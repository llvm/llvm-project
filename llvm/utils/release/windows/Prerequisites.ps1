# Prerequisite discovery and installation for Windows release builds.

function Find-Wix314Bin {
    # Prefer WiX found on PATH before checking the standard install locations.
    $BinDirs = @(Get-Command candle -All -CommandType Application -ErrorAction SilentlyContinue |
        ForEach-Object { Split-Path -Parent $_.Source })
    $BinDirs += @(
        "${env:ProgramFiles(x86)}\WiX Toolset v3.14\bin"
        "$env:ProgramFiles\WiX Toolset v3.14\bin"
    )

    foreach ($BinDir in ($BinDirs | Select-Object -Unique)) {
        $CandlePath = Join-Path $BinDir 'candle.exe'
        $LightPath = Join-Path $BinDir 'light.exe'
        if (-not (Test-Path -LiteralPath $CandlePath -PathType Leaf) -or
            -not (Test-Path -LiteralPath $LightPath -PathType Leaf)) { continue }
        if ((Get-Item -LiteralPath $CandlePath).VersionInfo.FileVersion -notlike '3.14.*' -or
            (Get-Item -LiteralPath $LightPath).VersionInfo.FileVersion -notlike '3.14.*') {
            continue
        }
        try {
            & $CandlePath -? *> $null
            if ($LASTEXITCODE -ne 0) { continue }
            & $LightPath -? *> $null
            if ($LASTEXITCODE -eq 0) { return $BinDir }
        } catch {
            # Try the next WiX installation location.
        }
    }
    return $null
}

function Get-CMakeVersion {
    param([Parameter(Mandatory)][string]$Path)

    try {
        $VersionOutput = (& $Path --version 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0 -or
            $VersionOutput -notmatch '(?m)^cmake version (\d+(?:\.\d+){1,3})\s*$') {
            return $null
        }
        return [version]$Matches[1]
    } catch {
        return $null
    }
}

function Find-CMakeExecutable {
    # Prefer the first usable version on PATH, then the standard CMake
    # installation, and finally CMake bundled with any Visual Studio instance.
    $Candidates = @()
    foreach ($PathDir in ($env:PATH -split ';')) {
        if ($PathDir) {
            $Candidates += Join-Path $PathDir 'cmake.exe'
        }
    }

    if ($env:ProgramFiles) {
        $Candidates += Join-Path $env:ProgramFiles 'CMake\bin\cmake.exe'
    }

    $VsInstallPaths = @()
    if ($env:VSINSTALLDIR) {
        $VsInstallPaths += $env:VSINSTALLDIR
    }
    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $VsWhere -PathType Leaf) {
        try {
            $VsInstallPaths += @(& $VsWhere -all -products '*' -property installationPath 2>$null)
        } catch {
            # CMake may still be available through PATH or the standard install.
        }
    }
    foreach ($VsInstallPath in ($VsInstallPaths | Where-Object { $_ } | Select-Object -Unique)) {
        $Candidates += Join-Path $VsInstallPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    }

    foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }

        $Version = Get-CMakeVersion -Path $Candidate
        if ($null -eq $Version -or $Version -lt $script:MinimumCMakeVersion) {
            if ($null -ne $Version) {
                Write-SubStep "Ignoring CMake $Version at $Candidate; LLVM requires $($script:MinimumCMakeVersion) or newer."
            }
            continue
        }

        # Put the selected directory first so every later `cmake` invocation
        # uses the validated executable, even if an older one appeared earlier.
        $BinDir = Split-Path -Parent $Candidate
        $OtherPaths = @($env:PATH -split ';' | Where-Object {
            $_ -and $_.TrimEnd('\') -ine $BinDir.TrimEnd('\')
        })
        $env:PATH = (@($BinDir) + $OtherPaths) -join ';'
        Write-SubStep "CMake ${version}: $Candidate"
        return $Candidate
    }

    return $null
}

function Find-SevenZipExecutable {
    $Candidates = @()
    $Command = Get-Command 7z -CommandType Application -ErrorAction SilentlyContinue
    if ($Command) { $Candidates += $Command.Source }
    foreach ($ProgramFiles in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($ProgramFiles) {
            $Candidates += Join-Path $ProgramFiles '7-Zip\7z.exe'
        }
    }

    foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }
        try {
            & $Candidate -? *> $null
            if ($LASTEXITCODE -ne 0) { continue }
        } catch {
            continue
        }

        # An existing installation may not have added 7z.exe to PATH.
        if (-not $Command -or $Command.Source -ne $Candidate) {
            $BinDir = Split-Path -Parent $Candidate
            $OtherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $BinDir })
            $env:PATH = (@($BinDir) + $OtherPaths) -join ';'
            Write-SubStep "Using 7-Zip from: $BinDir"
        }
        return $Candidate
    }
    return $null
}

function Find-MakeExecutable {
    $Candidates = @()
    $Command = Get-Command make -CommandType Application -ErrorAction SilentlyContinue
    if ($Command) { $Candidates += $Command.Source }
    foreach ($ProgramFiles in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
        if ($ProgramFiles) {
            $Candidates += Join-Path $ProgramFiles 'GnuWin32\bin\make.exe'
        }
    }

    foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf) -or
            -not (Test-ExecutableVersion -Path $Candidate)) { continue }

        if (-not $Command -or $Command.Source -ne $Candidate) {
            $BinDir = Split-Path -Parent $Candidate
            $OtherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $BinDir })
            $env:PATH = (@($BinDir) + $OtherPaths) -join ';'
            Write-SubStep "Using GNU Make from: $BinDir"
        }
        return $Candidate
    }
    return $null
}

function Find-OfficialLlvmBin {
    $BinDir = Join-Path $env:ProgramFiles 'LLVM\bin'
    $ClangCl = Join-Path $BinDir 'clang-cl.exe'
    $LldLink = Join-Path $BinDir 'lld-link.exe'
    if ((Test-Path -LiteralPath $ClangCl -PathType Leaf) -and
        (Test-Path -LiteralPath $LldLink -PathType Leaf) -and
        (Test-ExecutableVersion -Path $ClangCl) -and
        (Test-ExecutableVersion -Path $LldLink)) {
        return $BinDir
    }
    return $null
}

function Get-MissingPrerequisites {
    <#
    .SYNOPSIS
        Applies PATH fixups, checks required build tools, and returns the
        names of any that are missing.
    #>
    Write-Step "Validating prerequisites"

    # Fix up PATH for tools that install to well-known non-PATH locations.
    $PathFixups = @(
        "$env:ProgramFiles\Git\usr\bin"                     # sh, bash, sed, grep (needed by tests)
        "${env:ProgramFiles(x86)}\GnuWin32\bin"             # make (GnuWin32 default)
        "$env:ProgramFiles\GnuWin32\bin"                    # make (alt location)
    )
    foreach ($Dir in $PathFixups) {
        if ((Test-Path $Dir) -and ($env:PATH -notlike "*$Dir*")) {
            $env:PATH = "$env:PATH;$Dir"
            Write-SubStep "Added to PATH: $Dir"
        }
    }

    $CMakePath = Find-CMakeExecutable

    # CPack finds WiX through PATH. An older candle/light pair can appear
    # before the installed 3.14 tools (for example, in an Unreal SDK).
    $WixBin = Find-Wix314Bin
    if ($WixBin) {
        $OtherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $WixBin })
        $env:PATH = (@($WixBin) + $OtherPaths) -join ';'
        Write-SubStep "Using WiX 3.14 from: $WixBin"
    }

    # Check all required tools.
    $Required = @(
        @{ Name = "Ninja";      Cmd = "ninja" }
        @{ Name = "Python";     Cmd = "python" }
        @{ Name = "7-Zip";      Cmd = "7z" }
        @{ Name = "WiX candle"; Cmd = "candle" }
        @{ Name = "WiX light";  Cmd = "light" }
        @{ Name = "Git";        Cmd = "git" }
        @{ Name = "SWIG";       Cmd = "swig" }
        @{ Name = "Perl";       Cmd = "perl" }
        @{ Name = "Make";       Cmd = "make" }
    )

    $Missing = @()
    foreach ($Tool in $Required) {
        $Cmd = Get-Command $Tool.Cmd -ErrorAction SilentlyContinue
        $ToolPath = if ($Cmd) { $Cmd.Source } else { $null }
        # Windows may expose a python.exe App Execution Alias that is not an
        # installed interpreter.
        if ($Cmd -and $Tool.Cmd -eq 'python' -and
            -not (Test-ExecutableVersion -Path $Cmd.Source)) {
            $ToolPath = $null
        }
        if ($Tool.Cmd -eq '7z') {
            $ToolPath = Find-SevenZipExecutable
        }
        if ($Tool.Cmd -eq 'make') {
            $ToolPath = Find-MakeExecutable
        }
        # These tools are deprecated and removed in newer version of WiX.
        if ($Tool.Cmd -in @('candle', 'light') -and -not $WixBin) {
            $ToolPath = $null
        }
        if ($ToolPath) {
            Write-SubStep "$($Tool.Name): $ToolPath"
        } else {
            $Missing += $Tool.Name
        }
    }

    if (-not $CMakePath) {
        $Missing += "CMake $($script:MinimumCMakeVersion) or newer"
    }

    if (-not $ForceMSVC) {
        $LLVMBin = Find-OfficialLlvmBin
        if ($LLVMBin) {
            Write-SubStep "LLVM release (clang-cl and lld-link): $LLVMBin"
        } else {
            $Missing += 'LLVM release (clang-cl and lld-link)'
        }
    }

    # Lit disables the Python user site when running tests. Check the Python
    # installation itself so timeouts and tests requiring them stay available.
    $HostPython = Get-PythonExecutableForArch -Arch 'amd64'
    if ($HostPython) {
        if (Test-PythonPsutil -PythonExecutable $HostPython) {
            Write-SubStep "Python psutil: $HostPython"
        } else {
            $Missing += 'Python psutil'
        }
    }
    if ($script:BuildArch -eq 'arm64') {
        $Arm64Python = Get-PythonExecutableForArch -Arch 'arm64'
        if ($Arm64Python -and $Arm64Python -ne $HostPython) {
            if (Test-PythonPsutil -PythonExecutable $Arm64Python) {
                Write-SubStep "Python psutil (ARM64): $Arm64Python"
            } else {
                $Missing += 'Python psutil (ARM64)'
            }
        }
    }

    if (Test-NetFx3Installed) {
        Write-SubStep 'NetFx3 (.NET Framework 3.5): installed'
    } else {
        $Missing += 'NetFx3 (.NET Framework 3.5)'
    }

    return $Missing
}

function Assert-SevenZipSymlinkSupport {
    # 7-Zip 21+ extracts symlinks from LLVM's source archive. Windows must
    # allow symlink creation when -DownloadSource is selected.
    $Banner = (& 7z 2>&1 | Select-Object -First 2) -join ' '
    if ($Banner -notmatch '7-Zip\s+(\d+)\.' -or [int]$Matches[1] -lt 21) {
        return
    }

    $LinkPath = Join-Path ([IO.Path]::GetTempPath()) ("llvm-release-" + [guid]::NewGuid())
    try {
        New-Item -ItemType SymbolicLink -Path $LinkPath -Target $PWD.Path -ErrorAction Stop | Out-Null
    } catch {
        throw "7-Zip 21+ needs permission to create symlinks when extracting the source archive. Run elevated, enable Developer Mode, or use 7-Zip 20.x."
    } finally {
        Remove-Item -LiteralPath $LinkPath -Force -ErrorAction SilentlyContinue
    }
}

#===============================================================================
# Prerequisite installation (for fresh VMs)
#===============================================================================

function Test-IsElevated {
    $Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $Principal = [Security.Principal.WindowsPrincipal]::new($Identity)
    return $Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-NetFx3Installed {
    # The .NET Framework 3.5 installer records this value on both Windows
    # Server and desktop Windows. WiX 3.14 needs the feature to run.
    $Key = 'HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP\v3.5'
    $Framework = Get-ItemProperty -LiteralPath $Key -ErrorAction SilentlyContinue
    return $null -ne $Framework -and $Framework.Install -eq 1
}

function Install-NetFx3 {
    if (Test-NetFx3Installed) {
        Write-SubStep 'NetFx3 (.NET Framework 3.5) is already installed.'
        return
    }

    Write-SubStep 'Installing NetFx3 (.NET Framework 3.5)...'
    $InstallationType = (Get-ItemProperty `
        -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').InstallationType
    if ($InstallationType -like 'Server*') {
        # Windows Server exposes NetFx3 as the NET-Framework-Core server role.
        # Use the built-in Windows PowerShell to load ServerManager reliably,
        # even when this release script is running under PowerShell 7.
        $WindowsPowerShell = Join-Path $env:SystemRoot `
            'System32\WindowsPowerShell\v1.0\powershell.exe'
        $ServerCommand = @'
$ErrorActionPreference = 'Stop'
Import-Module ServerManager
$Feature = Get-WindowsFeature -Name NET-Framework-Core
if (-not $Feature) { throw 'NET-Framework-Core is not available on this server.' }
if (-not $Feature.Installed) {
    $Result = Install-WindowsFeature -Name NET-Framework-Core
    if (-not $Result.Success) {
        throw "Install-WindowsFeature failed: $($Result.ExitCode)"
    }
    if ($Result.RestartNeeded -eq 'Yes') { exit 3010 }
}
'@
        $EncodedCommand = [Convert]::ToBase64String(
            [Text.Encoding]::Unicode.GetBytes($ServerCommand))
        & $WindowsPowerShell -NoProfile -NonInteractive -EncodedCommand $EncodedCommand
        if ($LASTEXITCODE -eq 3010) {
            throw 'NetFx3 installation requires a restart. Restart Windows and rerun the release script.'
        }
        if ($LASTEXITCODE -ne 0) {
            throw "NetFx3 installation failed (exit code $LASTEXITCODE). If Windows cannot find the feature files, install NET-Framework-Core from matching Windows Server media and rerun the release script."
        }
    } else {
        $Result = Enable-WindowsOptionalFeature -Online -FeatureName NetFx3 `
            -All -NoRestart -ErrorAction Stop
        if ($Result.RestartNeeded) {
            throw 'NetFx3 installation requires a restart. Restart Windows and rerun the release script.'
        }
    }

    if (-not (Test-NetFx3Installed)) {
        throw 'NetFx3 installation completed, but .NET Framework 3.5 is not registered as installed.'
    }
    Write-SubStep 'NetFx3 installed.'
}

function Update-PathAfterPrerequisites {
    # The elevated installer runs in a separate process, so refresh the
    # caller's PATH from the machine and user environment after it exits.
    $MachinePath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
    $UserPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
    $env:PATH = "$MachinePath;$UserPath"
}

function Invoke-ElevatedPrerequisiteInstallation {
    $ReleaseRoot = $script:ReleaseScriptRoot.Replace("'", "''")
    $CommonPath = (Join-Path $script:ReleaseScriptRoot 'windows\Common.ps1').Replace("'", "''")
    $PrerequisitesPath = (Join-Path $script:ReleaseScriptRoot 'windows\Prerequisites.ps1').Replace("'", "''")
    $ToolchainPath = (Join-Path $script:ReleaseScriptRoot 'windows\Toolchain.ps1').Replace("'", "''")
    $BuildArch = $script:BuildArch.Replace("'", "''")
    $MinimumCMakeVersion = $script:MinimumCMakeVersion.ToString()
    $ForceMSVCValue = if ($ForceMSVC) { '$true' } else { '$false' }
    $UnattendedValue = if ($Unattended) { '$true' } else { '$false' }

    # Keep the elevated window open so the user can review the result.
    $PauseCommand = if (-not $Unattended) {
        "Read-Host 'Press Enter to close this window' | Out-Null"
    } else {
        ''
    }
    $WorkerCommand = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Stop'
`$script:ReleaseScriptRoot = '$ReleaseRoot'
`$script:MinimumCMakeVersion = [version]'$MinimumCMakeVersion'
`$script:BuildArch = '$BuildArch'
`$ForceMSVC = $ForceMSVCValue
`$Unattended = $UnattendedValue
. '$CommonPath'
. '$PrerequisitesPath'
. '$ToolchainPath'

`$ExitCode = 0
try {
    Install-Prerequisites
} catch {
    Write-Host "Prerequisite installation failed:" -ForegroundColor Red
    Write-Host (`$_ | Out-String) -ForegroundColor Red
    `$ExitCode = 1
}
$PauseCommand
exit `$ExitCode
"@
    $EncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($WorkerCommand))

    Write-SubStep 'Requesting administrator access to install prerequisites...'
    try {
        $Process = Start-Process -FilePath (Join-Path $PSHOME 'pwsh.exe') `
            -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $EncodedCommand) `
            -Verb RunAs -Wait -PassThru
    } catch {
        throw "Could not start elevated prerequisite installation: $_"
    }
    if ($Process.ExitCode -ne 0) {
        throw "Elevated prerequisite installation failed (exit code $($Process.ExitCode))."
    }
}

function Find-WingetExecutable {
    $Candidates = @()
    $Winget = Get-Command winget -CommandType Application -ErrorAction SilentlyContinue
    if ($Winget) { $Candidates += $Winget.Source }
    if ($env:LOCALAPPDATA) {
        $Candidates += Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\winget.exe'
    }

    # On fresh Windows Server images the App Installer alias may be absent
    # from the elevated PATH. Discover the installed package directly too.
    try {
        $Packages = @(Get-AppxPackage -AllUsers -Name Microsoft.DesktopAppInstaller -ErrorAction Stop)
        foreach ($Package in ($Packages | Sort-Object Version -Descending)) {
            if ($Package.InstallLocation) {
                $Candidates += Join-Path $Package.InstallLocation 'winget.exe'
            }
        }
    } catch {
        # Appx discovery is not available on every Windows image.
    }

    foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }
        try {
            & $Candidate --version *> $null
            if ($LASTEXITCODE -eq 0) { return $Candidate }
        } catch {
            # Try the next location.
        }
    }
    throw "winget cannot run in the elevated shell. Checked: $($Candidates -join ', '). Ensure App Installer is available to the account used by UAC."
}

function Assert-WixToolsUsable {
    $BinDir = Find-Wix314Bin
    if (-not $BinDir) {
        throw 'WiX 3.14 was installed, but candle.exe and light.exe could not run.'
    }
    Write-SubStep "WiX tools verified: $BinDir"
}

function Install-Prerequisites {
    if (-not (Test-IsElevated)) {
        if ($Unattended) {
            throw "Unattended prerequisite installation requires administrator privileges. Start an elevated shell before using -Unattended -InstallPrerequisites."
        }
        Invoke-ElevatedPrerequisiteInstallation
        Update-PathAfterPrerequisites
        if (-not (Find-CMakeExecutable)) {
            throw "CMake $($script:MinimumCMakeVersion) or newer could not be found after prerequisite installation."
        }
        Install-PythonPsutilForRequestedArchitectures
        return
    }

    Write-Step "Installing prerequisites"

    Install-NetFx3
    $WingetPath = Find-WingetExecutable

    # Version requirements for the release build:
    #   - CMake 3.31 or newer
    #   - NetFx3 (.NET Framework 3.5) for WiX Toolset 3.14
    #   - Python 3.13 for all architectures
    #   - SWIG 4 or newer for LLDB (install the latest available WinGet version)
    #   - WiX Toolset 3.14 for MSI packaging
    #   - Perl is needed for the OpenMP run-time
    #   - GNU Make is needed by LLDB API tests
    #   - 7-Zip 21.x+ needs permission to create source archive symlinks
    #   - LLVM.LLVM supplies the stage 0 clang-cl and lld-link tools
    $Tools = @(
        @{ Name = "cmake";      WingetId = "Kitware.CMake";                 Version = "";            Verify = "cmake" }
        @{ Name = "ninja";      WingetId = "Ninja-build.Ninja";             Version = "";            Verify = "ninja" }
        @{ Name = "python3.13"; WingetId = "Python.Python.3.13";            Version = "";            Verify = "python" }
        @{ Name = "7z";         WingetId = "7zip.7zip";                     Version = "";            Verify = "7z" }
        @{ Name = "wix";        WingetId = "WiXToolset.WiXToolset";         Version = "3.14.1.8722"; Verify = "candle" }
        @{ Name = "git";        WingetId = "Git.Git";                       Version = "";            Verify = "git" }
        @{ Name = "swig";       WingetId = "SWIG.SWIG";                     Version = "";            Verify = "swig" }
        @{ Name = "perl";       WingetId = "StrawberryPerl.StrawberryPerl"; Version = "";            Verify = "perl" }
        @{ Name = "make";       WingetId = "GnuWin32.Make";                 Version = "";            Verify = "make" }
    )

    foreach ($Tool in $Tools) {
        $Cmd = Get-Command $Tool.Verify -ErrorAction SilentlyContinue
        if ($Cmd -and $Tool.Verify -eq 'python' -and
            -not (Test-ExecutableVersion -Path $Cmd.Source)) {
            $Cmd = $null
        }
        $ExistingPath = if ($Tool.Name -eq 'cmake') { Find-CMakeExecutable }
            elseif ($Tool.Name -eq 'wix') { Find-Wix314Bin }
            elseif ($Tool.Name -eq '7z') { Find-SevenZipExecutable }
            elseif ($Tool.Name -eq 'make') { Find-MakeExecutable }
            elseif ($Cmd) { $Cmd.Source }
        if ($ExistingPath) {
            Write-SubStep "$($Tool.Name) is already installed: $ExistingPath"
        } else {
            $VersionArg = if ($Tool.Version) { @("--version", $Tool.Version) } else { @() }
            $VersionLabel = if ($Tool.Version) { " $($Tool.Version)" } else { '' }
            Write-SubStep "Installing $($Tool.Name) ($($Tool.WingetId)$VersionLabel)..."
            $InstallArgs = @('install', '--id', $Tool.WingetId) + $VersionArg +
                @('--accept-package-agreements', '--accept-source-agreements', '--silent')
            & $WingetPath @InstallArgs
            $InstallExitCode = $LASTEXITCODE
            if ($Tool.Name -eq 'wix' -and $InstallExitCode -eq -1978334960) {
                # winget can fail while checking its NetFx3 dependency on
                # Server 2025 even after the feature is installed. The feature
                # was installed and verified above; bypass only this check.
                Write-Warning 'winget could not verify the installed NetFx3 feature. Retrying WiX with --force.'
                & $WingetPath @InstallArgs --force
                $InstallExitCode = $LASTEXITCODE
            }
            if ($Tool.Name -eq '7z') {
                $SevenZipPath = Find-SevenZipExecutable
                if ($SevenZipPath) {
                    if ($InstallExitCode -ne 0) {
                        Write-Warning "winget exited with code $InstallExitCode, but 7-Zip is usable at $SevenZipPath. Continuing."
                    }
                    Write-SubStep "7-Zip verified: $SevenZipPath"
                    continue
                }
            }
            if ($Tool.Name -eq 'make') {
                $MakePath = Find-MakeExecutable
                if ($MakePath) {
                    if ($InstallExitCode -ne 0) {
                        Write-Warning "winget exited with code $InstallExitCode, but GNU Make is usable at $MakePath. Continuing."
                    }
                    Write-SubStep "GNU Make verified: $MakePath"
                    continue
                }
            }
            if ($Tool.Name -eq 'cmake') {
                if (-not (Find-CMakeExecutable)) {
                    throw "CMake installation finished, but version $($script:MinimumCMakeVersion) or newer could not be found."
                }
                continue
            }
            if ($InstallExitCode -ne 0) {
                if ($Tool.Name -eq 'wix') {
                    throw "Failed to install WiX via winget (exit code $InstallExitCode). NetFx3 is installed; inspect winget's error above."
                }
                throw "Failed to install $($Tool.Name) via winget (exit code $InstallExitCode)."
            }
            if ($Tool.Name -eq '7z') {
                throw '7-Zip installation finished, but 7z.exe could not be found or run.'
            }
            if ($Tool.Name -eq 'make') {
                throw 'GNU Make installation finished, but make.exe could not be found or run.'
            }
            if ($Tool.Name -eq 'wix') { Assert-WixToolsUsable }
        }
    }

    if (-not $ForceMSVC) {
        $LLVMBin = Find-OfficialLlvmBin
        if ($LLVMBin) {
            Write-SubStep "LLVM release is already installed: $LLVMBin"
        } else {
            Write-SubStep 'Installing LLVM release (LLVM.LLVM)...'
            & $WingetPath install --id LLVM.LLVM --exact --source winget `
                --accept-package-agreements --accept-source-agreements --silent
            $InstallExitCode = $LASTEXITCODE
            $LLVMBin = Find-OfficialLlvmBin
            if (-not $LLVMBin) {
                throw "LLVM installation did not provide usable clang-cl.exe and lld-link.exe in $env:ProgramFiles\LLVM\bin (winget exit code $InstallExitCode)."
            }
            if ($InstallExitCode -ne 0) {
                Write-Warning "winget exited with code $InstallExitCode, but LLVM tools are usable at $LLVMBin. Continuing."
            }
            Write-SubStep "LLVM release verified: $LLVMBin"
        }
    }

    # Install ARM64 Python when building ARM64 so LLDB uses a matching Python.
    # The main Python install above covers the runner's default architecture.
    $ArchPython = @()
    if ($script:BuildArch -eq 'arm64') { $ArchPython += @{ Arch = 'arm64'; Suffix = '-arm64' } }
    foreach ($Ap in $ArchPython) {
        $BasePath = "$env:LOCALAPPDATA\Programs\Python"
        $Existing = if (Test-Path $BasePath) {
            Get-ChildItem -Path $BasePath -Directory -Filter "Python3*$($Ap.Suffix)" |
                Where-Object { Test-Path (Join-Path $_.FullName 'python.exe') }
        } else { $null }
        if ($Existing) {
            Write-SubStep "Python for $($Ap.Arch) already installed: $($Existing[0].FullName)"
        } else {
            Write-SubStep "Installing Python for $($Ap.Arch) (Python.Python.3.13 --architecture $($Ap.Arch))..."
            & $WingetPath install --id Python.Python.3.13 --architecture $($Ap.Arch) --accept-package-agreements --accept-source-agreements --silent
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Failed to install Python for $($Ap.Arch). LLDB may not work for this architecture."
            }
        }
    }

    # Visual Studio Build Tools - special handling
    Install-VisualStudio -WingetPath $WingetPath

    # Get-MissingPrerequisites will add non-standard paths such as GnuWin32.
    Update-PathAfterPrerequisites
    if (-not (Find-CMakeExecutable)) {
        throw "CMake $($script:MinimumCMakeVersion) or newer could not be found after prerequisite installation."
    }
    Install-PythonPsutilForRequestedArchitectures
}

function Install-VisualStudio {
    param([Parameter(Mandatory)][string]$WingetPath)

    # Components required by LLVM:
    #   - VCTools workload: core C++ compiler, linker, libs
    #   - ATL: required by llvm/include/llvm/DebugInfo/PDB/DIA/DIASupport.h (<atlbase.h>)
    #   - DIA SDK: PDB debug info reader, checked by cmake/config-ix.cmake (LLVM_ENABLE_DIA_SDK)
    #     The DIA SDK is part of the "Visual Studio C++ core features" component.
    #
    # Component IDs reference:
    #   https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools

    $RequiredComponents = @(
        "Microsoft.VisualStudio.Workload.VCTools"
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
        "Microsoft.VisualStudio.Component.VC.Tools.ARM64"
        "Microsoft.VisualStudio.Component.VC.ATL"
        "Microsoft.VisualStudio.Component.VC.ATL.ARM64"
        "Microsoft.VisualStudio.Component.VC.DiagnosticTools"  # includes DIA SDK
        # The latest Windows SDK is included automatically via --includeRecommended
    )

    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $VsWhere) {
        $Vs = & $VsWhere -nologo -latest -products '*' -format json | ConvertFrom-Json
        if ($Vs) {
            Write-SubStep "Visual Studio already installed: $($Vs[0].installationPath)"
            Write-SubStep "Ensuring required components are installed..."
            # setup.exe modify does not accept --wait; wait for the process
            # from PowerShell instead.
            $AddArgs = @()
            foreach ($Component in $RequiredComponents) {
                $AddArgs += '--add'
                $AddArgs += $Component
            }
            $Installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\setup.exe"
            if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
                throw "Visual Studio Installer not found: $Installer"
            }
            $InstallPath = $Vs[0].installationPath
            $InstallArgs = @('modify', '--installPath', "`"$InstallPath`"") +
                $AddArgs + @('--includeRecommended', '--passive', '--norestart')
            $Process = Start-Process -FilePath $Installer -ArgumentList $InstallArgs -Wait -PassThru
            if ($Process.ExitCode -eq 3010) {
                throw 'Visual Studio components were installed, but Windows must be restarted before building.'
            }
            if ($Process.ExitCode -ne 0) {
                throw "Failed to modify Visual Studio installation (exit code $($Process.ExitCode))."
            }
            return
        }
    }

    Write-SubStep "Installing Visual Studio Build Tools 2022..."
    $AddArgs = ($RequiredComponents | ForEach-Object { "--add $_" }) -join ' '
    & $WingetPath install Microsoft.VisualStudio.2022.BuildTools `
        --accept-package-agreements --accept-source-agreements --silent `
        --override "$AddArgs --includeRecommended --passive --wait"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Visual Studio Build Tools."
    }
}

