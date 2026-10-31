# Visual Studio, Python, and SWIG setup for Windows release builds.

function Find-VisualStudio {
    <#
    .SYNOPSIS
        Finds the Visual Studio installation and returns the path to VsDevCmd.bat.
    #>

    if ($env:VSINSTALLDIR) {
        Write-SubStep "Using enabled Visual Studio installation: $env:VSINSTALLDIR"
        $VsInstall = $env:VSINSTALLDIR
    } else {
        $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path $VsWhere)) {
            throw "Cannot find vswhere.exe. Is Visual Studio installed?"
        }
        $VsInstall = & $VsWhere -nologo -latest -products '*' -all -property installationPath
        if (-not $VsInstall) {
            throw "Cannot find any Visual Studio installation."
        }
        Write-SubStep "Detected Visual Studio: $VsInstall"
    }

    $VsDevCmd = Join-Path $VsInstall 'Common7' 'Tools' 'VsDevCmd.bat'
    if (-not (Test-Path $VsDevCmd)) {
        throw "Cannot find VsDevCmd.bat at: $VsDevCmd"
    }
    return $VsDevCmd
}

function Enter-VsDevEnvironment {
    <#
    .SYNOPSIS
        Sources VsDevCmd.bat and imports the resulting environment variables into PowerShell.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$VsDevCmd,
        [Parameter(Mandatory)]
        [string]$Arch
    )
    Write-SubStep "Setting up VS developer environment for $Arch..."
    # Run VsDevCmd.bat in a cmd subprocess and capture the resulting environment
    $EnvBlock = cmd /c "`"$VsDevCmd`" -arch=$Arch -no_logo && set" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "VsDevCmd.bat failed for $Arch (exit code $LASTEXITCODE)."
    }
    # Reimport the environment variables from the previous command into the current shell.
    foreach ($Line in $EnvBlock) {
        if ($Line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
}

function Use-DiaSdkRuntime {
    param([Parameter(Mandatory)][string]$Arch)

    if (-not $env:VSINSTALLDIR) {
        throw 'VsDevCmd did not set VSINSTALLDIR; cannot locate the DIA SDK runtime.'
    }
    # LLVM loads msdia140.dll by name when COM registration is unavailable.
    # Put the target architecture's DLL before other copies on PATH so an x86
    # DLL cannot be loaded by the x64 or arm64 test tools.
    $DiaBin = Join-Path $env:VSINSTALLDIR "DIA SDK\bin\$Arch"
    $DiaDll = Join-Path $DiaBin 'msdia140.dll'
    if (-not (Test-Path -LiteralPath $DiaDll -PathType Leaf)) {
        throw "DIA runtime not found: $DiaDll. Install the Visual Studio Diagnostic Tools component."
    }
    $OtherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $DiaBin })
    $env:PATH = (@($DiaBin) + $OtherPaths) -join ';'
    Write-SubStep "Using DIA runtime: $DiaDll"
}

#===============================================================================
# Python setup
#===============================================================================

function Test-PythonPsutil {
    param([Parameter(Mandatory)][string]$PythonExecutable)

    try {
        # Lit clears PYTHONPATH and disables the user site. -I performs the
        # same checks and also avoids accidentally importing from the source tree.
        & $PythonExecutable -I -c 'import psutil' *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Install-PythonPsutil {
    param([Parameter(Mandatory)][string]$PythonExecutable)

    if (Test-PythonPsutil -PythonExecutable $PythonExecutable) {
        Write-SubStep "Python psutil is already installed: $PythonExecutable"
        return
    }

    Write-SubStep "Installing Python psutil into $PythonExecutable..."
    & $PythonExecutable -I -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        & $PythonExecutable -I -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) {
            throw "Could not bootstrap pip for $PythonExecutable."
        }
    }
    & $PythonExecutable -I -m pip install --disable-pip-version-check --no-input psutil
    if ($LASTEXITCODE -ne 0 -or -not (Test-PythonPsutil -PythonExecutable $PythonExecutable)) {
        throw "Could not install psutil into $PythonExecutable. Lit requires it for test timeouts."
    }
    Write-SubStep "Python psutil verified: $PythonExecutable"
}

function Install-PythonPsutilForRequestedArchitectures {
    $HostPython = Get-PythonExecutableForArch -Arch 'amd64'
    if (-not $HostPython) {
        throw 'Python was installed, but its executable could not be found to install psutil.'
    }
    $PythonExecutables = @($HostPython)
    if ($script:BuildArch -eq 'arm64') {
        $Arm64Python = Get-PythonExecutableForArch -Arch 'arm64'
        if ($Arm64Python) { $PythonExecutables += $Arm64Python }
    }
    foreach ($PythonExecutable in ($PythonExecutables | Select-Object -Unique)) {
        Install-PythonPsutil -PythonExecutable $PythonExecutable
    }
}

function Get-PythonInstallDir {
    param([Parameter(Mandatory)][string]$PythonCommand)

    # Python Manager and winget can put only a shim in PATH. CMake needs the
    # actual installation directory, which sys.exec_prefix reports.
    $Prefix = & $PythonCommand -c 'import sys; print(sys.exec_prefix)'
    if ($LASTEXITCODE -ne 0 -or -not $Prefix) {
        throw "Could not find the Python installation behind $PythonCommand"
    }
    $InstallDir = ([string]$Prefix).Trim()
    if (-not (Test-Path (Join-Path $InstallDir 'python.exe'))) {
        throw "Python reported $InstallDir, but python.exe is missing there."
    }
    return $InstallDir
}

function Test-PythonIsArm64 {
    param([Parameter(Mandatory)][string]$PythonExecutable)

    try {
        $Machine = & $PythonExecutable -c 'import platform; print(platform.machine())'
        return $LASTEXITCODE -eq 0 -and ([string]$Machine).Trim() -eq 'ARM64'
    } catch {
        return $false
    }
}

function Get-PythonExecutableForArch {
    param([Parameter(Mandatory)][ValidateSet('amd64', 'arm64')][string]$Arch)

    if ($Arch -eq 'amd64') {
        $Python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $Python -or -not (Test-ExecutableVersion -Path $Python.Source)) {
            return $null
        }
        try {
            $PythonHome = Get-PythonInstallDir -PythonCommand $Python.Source
            return Join-Path $PythonHome 'python.exe'
        } catch {
            return $null
        }
    }

    $BasePath = "$env:LOCALAPPDATA\Programs\Python"
    if (Test-Path $BasePath) {
        $Candidates = @(Get-ChildItem -Path $BasePath -Directory -Filter 'Python3*-arm64' |
            Where-Object { Test-Path (Join-Path $_.FullName 'python.exe') } |
            Sort-Object Name -Descending)
        if ($Candidates.Count -gt 0) {
            return Join-Path $Candidates[0].FullName 'python.exe'
        }
    }
    return $null
}

function Find-Python {
    <#
    .SYNOPSIS
        Locates a Python installation for the given target architecture.

        For x64 (amd64) builds, uses the system Python from PATH.
        For ARM64 builds, probes standard install locations for any ARM64
        Python 3.x. Falls back to the PATH Python with a warning if no
        ARM64 install is found.
    #>
    param(
        [Parameter(Mandatory)][string]$Arch
    )

    $PythonHome = $null

    if ($Arch -eq 'amd64') {
        # Host-arch build: use system Python from PATH.
        $PythonExecutable = Get-PythonExecutableForArch -Arch 'amd64'
        if (-not $PythonExecutable) {
            throw "Cannot find python in PATH. Run with -InstallPrerequisites or install Python manually."
        }
        $PythonHome = Split-Path -Parent $PythonExecutable
    } else {
        # ARM64 build: probe the standard ARM64 Python install location.
        # Python installs to %LOCALAPPDATA%\Programs\Python\Python3XX-arm64
        $Suffix = '-arm64'
        $BasePath = "$env:LOCALAPPDATA\Programs\Python"
        $PythonExecutable = Get-PythonExecutableForArch -Arch 'arm64'
        if ($PythonExecutable) { $PythonHome = Split-Path -Parent $PythonExecutable }

        if (-not $PythonHome) {
            # Fall back to PATH Python with a warning.
            $PythonExecutable = Get-PythonExecutableForArch -Arch 'amd64'
            if ($PythonExecutable) {
                $PythonHome = Split-Path -Parent $PythonExecutable
                # On a native ARM64 machine (e.g. CI) the PATH Python is
                # already the right one.
                if (Test-PythonIsArm64 -PythonExecutable $PythonExecutable) {
                    Write-SubStep "PATH Python is already ARM64: $PythonHome"
                } else {
                    Write-Warning ("No per-arch Python found for $Arch (looked in $BasePath\Python3*$Suffix).`n" +
                        "  Falling back to PATH Python: $PythonHome`n" +
                        "  LLDB may not work correctly for the $Arch target.`n" +
                        "  Run with -InstallPrerequisites to install architecture-matched Python.")
                }
            } else {
                throw ("Cannot find Python for $Arch. No per-arch install in $BasePath and no python in PATH.`n" +
                    "Run with -InstallPrerequisites to install Python.")
            }
        }
    }

    # Display the version without returning it alongside the Python directory.
    Invoke-NativeCommand (Join-Path $PythonHome 'python.exe') --version | Out-Host
    $env:PYTHONHOME = $PythonHome
    $env:PATH = "$PythonHome;$env:PATH"
    Write-SubStep "Using Python from: $PythonHome"
    return $PythonHome
}

#===============================================================================
# SWIG setup
#===============================================================================

function Find-Swig {
    <#
    .SYNOPSIS
        Locates SWIG's real executable and library directory.
    #>
    $Command = Get-Command swig.exe -CommandType Application -ErrorAction Stop
    $Executable = $Command.Source
    $Item = Get-Item -LiteralPath $Executable -ErrorAction Stop

    # WinGet can put a symlink to swig.exe in PATH. SWIG may look for its
    # library beside that link rather than beside the installed executable.
    $LinkType = $Item.PSObject.Properties['LinkType']
    if ($LinkType -and $LinkType.Value -eq 'SymbolicLink') {
        $TargetProperty = $Item.PSObject.Properties['Target']
        $Target = if ($TargetProperty) { $TargetProperty.Value } else { $null }
        if ($Target -is [array]) { $Target = $Target[0] }
        if (-not $Target) {
            throw "Cannot resolve SWIG symlink: $Executable"
        }
        if (-not [IO.Path]::IsPathRooted($Target)) {
            $Target = Join-Path $Item.DirectoryName $Target
        }
        $Executable = (Resolve-Path -LiteralPath $Target -ErrorAction Stop).Path
    }

    $ReportedLibDir = Invoke-NativeCommand $Executable -swiglib
    if ($ReportedLibDir -is [array]) { $ReportedLibDir = $ReportedLibDir[-1] }
    $LibDir = ([string]$ReportedLibDir).Trim()
    if (-not $LibDir -or -not (Test-Path -LiteralPath (Join-Path $LibDir 'swig.swg') -PathType Leaf)) {
        $LibDir = Join-Path (Split-Path -Parent $Executable) 'Lib'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $LibDir 'swig.swg') -PathType Leaf)) {
        throw "SWIG library swig.swg was not found for $Executable. Check the SWIG installation."
    }

    Write-SubStep "Using SWIG: $Executable (library: $LibDir)"
    return @{ Executable = $Executable; LibraryDirectory = $LibDir }
}

