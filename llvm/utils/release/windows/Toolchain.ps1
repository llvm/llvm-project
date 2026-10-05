# Visual Studio, Python, and SWIG setup for Windows release builds.

function Find-VisualStudio {
    <#
    .SYNOPSIS
        Finds the Visual Studio installation and returns the path to VsDevCmd.bat.
    #>

    if ($env:VSINSTALLDIR) {
        Write-SubStep "Using enabled Visual Studio installation: $env:VSINSTALLDIR"
        $vsInstall = $env:VSINSTALLDIR
    } else {
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path $vswhere)) {
            throw "Cannot find vswhere.exe. Is Visual Studio installed?"
        }
        $vsInstall = & $vswhere -nologo -latest -products '*' -all -property installationPath
        if (-not $vsInstall) {
            throw "Cannot find any Visual Studio installation."
        }
        Write-SubStep "Detected Visual Studio: $vsInstall"
    }

    $vsDevCmd = Join-Path $vsInstall 'Common7' 'Tools' 'VsDevCmd.bat'
    if (-not (Test-Path $vsDevCmd)) {
        throw "Cannot find VsDevCmd.bat at: $vsDevCmd"
    }
    return $vsDevCmd
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
    $envBlock = cmd /c "`"$VsDevCmd`" -arch=$Arch -no_logo && set" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "VsDevCmd.bat failed for $Arch (exit code $LASTEXITCODE)."
    }
    # Reimport the environment variables from the previous command into the current shell.
    foreach ($line in $envBlock) {
        if ($line -match '^([^=]+)=(.*)$') {
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
    $diaBin = Join-Path $env:VSINSTALLDIR "DIA SDK\bin\$Arch"
    $diaDll = Join-Path $diaBin 'msdia140.dll'
    if (-not (Test-Path -LiteralPath $diaDll -PathType Leaf)) {
        throw "DIA runtime not found: $diaDll. Install the Visual Studio Diagnostic Tools component."
    }
    $otherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $diaBin })
    $env:PATH = (@($diaBin) + $otherPaths) -join ';'
    Write-SubStep "Using DIA runtime: $diaDll"
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
    $hostPython = Get-PythonExecutableForArch -Arch 'amd64'
    if (-not $hostPython) {
        throw 'Python was installed, but its executable could not be found to install psutil.'
    }
    $pythonExecutables = @($hostPython)
    if ($script:BuildArch -eq 'arm64') {
        $arm64Python = Get-PythonExecutableForArch -Arch 'arm64'
        if ($arm64Python) { $pythonExecutables += $arm64Python }
    }
    foreach ($pythonExecutable in ($pythonExecutables | Select-Object -Unique)) {
        Install-PythonPsutil -PythonExecutable $pythonExecutable
    }
}

function Get-PythonInstallDir {
    param([Parameter(Mandatory)][string]$PythonCommand)

    # Python Manager and winget can put only a shim in PATH. CMake needs the
    # actual installation directory, which sys.exec_prefix reports.
    $prefix = & $PythonCommand -c 'import sys; print(sys.exec_prefix)'
    if ($LASTEXITCODE -ne 0 -or -not $prefix) {
        throw "Could not find the Python installation behind $PythonCommand"
    }
    $installDir = ([string]$prefix).Trim()
    if (-not (Test-Path (Join-Path $installDir 'python.exe'))) {
        throw "Python reported $installDir, but python.exe is missing there."
    }
    return $installDir
}

function Test-PythonIsArm64 {
    param([Parameter(Mandatory)][string]$PythonExecutable)

    try {
        $machine = & $PythonExecutable -c 'import platform; print(platform.machine())'
        return $LASTEXITCODE -eq 0 -and ([string]$machine).Trim() -eq 'ARM64'
    } catch {
        return $false
    }
}

function Get-PythonExecutableForArch {
    param([Parameter(Mandatory)][ValidateSet('amd64', 'arm64')][string]$Arch)

    if ($Arch -eq 'amd64') {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python -or -not (Test-ExecutableVersion -Path $python.Source)) {
            return $null
        }
        try {
            $pythonHome = Get-PythonInstallDir -PythonCommand $python.Source
            return Join-Path $pythonHome 'python.exe'
        } catch {
            return $null
        }
    }

    $basePath = "$env:LOCALAPPDATA\Programs\Python"
    if (Test-Path $basePath) {
        $candidates = @(Get-ChildItem -Path $basePath -Directory -Filter 'Python3*-arm64' |
            Where-Object { Test-Path (Join-Path $_.FullName 'python.exe') } |
            Sort-Object Name -Descending)
        if ($candidates.Count -gt 0) {
            return Join-Path $candidates[0].FullName 'python.exe'
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

    $pythonHome = $null

    if ($Arch -eq 'amd64') {
        # Host-arch build: use system Python from PATH.
        $pythonExecutable = Get-PythonExecutableForArch -Arch 'amd64'
        if (-not $pythonExecutable) {
            throw "Cannot find python in PATH. Run with -InstallPrerequisites or install Python manually."
        }
        $pythonHome = Split-Path -Parent $pythonExecutable
    } else {
        # ARM64 build: probe the standard ARM64 Python install location.
        # Python installs to %LOCALAPPDATA%\Programs\Python\Python3XX-arm64
        $suffix = '-arm64'
        $basePath = "$env:LOCALAPPDATA\Programs\Python"
        $pythonExecutable = Get-PythonExecutableForArch -Arch 'arm64'
        if ($pythonExecutable) { $pythonHome = Split-Path -Parent $pythonExecutable }

        if (-not $pythonHome) {
            # Fall back to PATH Python with a warning.
            $pythonExecutable = Get-PythonExecutableForArch -Arch 'amd64'
            if ($pythonExecutable) {
                $pythonHome = Split-Path -Parent $pythonExecutable
                # On a native ARM64 machine (e.g. CI) the PATH Python is
                # already the right one.
                if (Test-PythonIsArm64 -PythonExecutable $pythonExecutable) {
                    Write-SubStep "PATH Python is already ARM64: $pythonHome"
                } else {
                    Write-Warning ("No per-arch Python found for $Arch (looked in $basePath\Python3*$suffix).`n" +
                        "  Falling back to PATH Python: $pythonHome`n" +
                        "  LLDB may not work correctly for the $Arch target.`n" +
                        "  Run with -InstallPrerequisites to install architecture-matched Python.")
                }
            } else {
                throw ("Cannot find Python for $Arch. No per-arch install in $basePath and no python in PATH.`n" +
                    "Run with -InstallPrerequisites to install Python.")
            }
        }
    }

    # Display the version without returning it alongside the Python directory.
    Invoke-NativeCommand (Join-Path $pythonHome 'python.exe') --version | Out-Host
    $env:PYTHONHOME = $pythonHome
    $env:PATH = "$pythonHome;$env:PATH"
    Write-SubStep "Using Python from: $pythonHome"
    return $pythonHome
}

#===============================================================================
# SWIG setup
#===============================================================================

function Find-Swig {
    <#
    .SYNOPSIS
        Locates SWIG's real executable and library directory.
    #>
    $command = Get-Command swig.exe -CommandType Application -ErrorAction Stop
    $executable = $command.Source
    $item = Get-Item -LiteralPath $executable -ErrorAction Stop

    # WinGet can put a symlink to swig.exe in PATH. SWIG may look for its
    # library beside that link rather than beside the installed executable.
    $linkType = $item.PSObject.Properties['LinkType']
    if ($linkType -and $linkType.Value -eq 'SymbolicLink') {
        $targetProperty = $item.PSObject.Properties['Target']
        $target = if ($targetProperty) { $targetProperty.Value } else { $null }
        if ($target -is [array]) { $target = $target[0] }
        if (-not $target) {
            throw "Cannot resolve SWIG symlink: $executable"
        }
        if (-not [IO.Path]::IsPathRooted($target)) {
            $target = Join-Path $item.DirectoryName $target
        }
        $executable = (Resolve-Path -LiteralPath $target -ErrorAction Stop).Path
    }

    $reportedLibDir = Invoke-NativeCommand $executable -swiglib
    if ($reportedLibDir -is [array]) { $reportedLibDir = $reportedLibDir[-1] }
    $libDir = ([string]$reportedLibDir).Trim()
    if (-not $libDir -or -not (Test-Path -LiteralPath (Join-Path $libDir 'swig.swg') -PathType Leaf)) {
        $libDir = Join-Path (Split-Path -Parent $executable) 'Lib'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $libDir 'swig.swg') -PathType Leaf)) {
        throw "SWIG library swig.swg was not found for $executable. Check the SWIG installation."
    }

    Write-SubStep "Using SWIG: $executable (library: $libDir)"
    return @{ Executable = $executable; LibraryDirectory = $libDir }
}

