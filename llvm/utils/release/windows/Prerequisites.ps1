# Prerequisite discovery and installation for Windows release builds.

function Find-Wix314Bin {
    # Prefer WiX found on PATH before checking the standard install locations.
    $binDirs = @(Get-Command candle -All -CommandType Application -ErrorAction SilentlyContinue |
        ForEach-Object { Split-Path -Parent $_.Source })
    $binDirs += @(
        "${env:ProgramFiles(x86)}\WiX Toolset v3.14\bin"
        "$env:ProgramFiles\WiX Toolset v3.14\bin"
    )

    foreach ($binDir in ($binDirs | Select-Object -Unique)) {
        $candlePath = Join-Path $binDir 'candle.exe'
        $lightPath = Join-Path $binDir 'light.exe'
        if (-not (Test-Path -LiteralPath $candlePath -PathType Leaf) -or
            -not (Test-Path -LiteralPath $lightPath -PathType Leaf)) { continue }
        if ((Get-Item -LiteralPath $candlePath).VersionInfo.FileVersion -notlike '3.14.*' -or
            (Get-Item -LiteralPath $lightPath).VersionInfo.FileVersion -notlike '3.14.*') {
            continue
        }
        try {
            & $candlePath -? *> $null
            if ($LASTEXITCODE -ne 0) { continue }
            & $lightPath -? *> $null
            if ($LASTEXITCODE -eq 0) { return $binDir }
        } catch {
            # Try the next WiX installation location.
        }
    }
    return $null
}

function Get-CMakeVersion {
    param([Parameter(Mandatory)][string]$Path)

    try {
        $versionOutput = (& $Path --version 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0 -or
            $versionOutput -notmatch '(?m)^cmake version (\d+(?:\.\d+){1,3})\s*$') {
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
    $candidates = @()
    foreach ($pathDir in ($env:PATH -split ';')) {
        if ($pathDir) {
            $candidates += Join-Path $pathDir 'cmake.exe'
        }
    }

    if ($env:ProgramFiles) {
        $candidates += Join-Path $env:ProgramFiles 'CMake\bin\cmake.exe'
    }

    $vsInstallPaths = @()
    if ($env:VSINSTALLDIR) {
        $vsInstallPaths += $env:VSINSTALLDIR
    }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        try {
            $vsInstallPaths += @(& $vswhere -all -products '*' -property installationPath 2>$null)
        } catch {
            # CMake may still be available through PATH or the standard install.
        }
    }
    foreach ($vsInstallPath in ($vsInstallPaths | Where-Object { $_ } | Select-Object -Unique)) {
        $candidates += Join-Path $vsInstallPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }

        $version = Get-CMakeVersion -Path $candidate
        if ($null -eq $version -or $version -lt $script:MinimumCMakeVersion) {
            if ($null -ne $version) {
                Write-SubStep "Ignoring CMake $version at $candidate; LLVM requires $($script:MinimumCMakeVersion) or newer."
            }
            continue
        }

        # Put the selected directory first so every later `cmake` invocation
        # uses the validated executable, even if an older one appeared earlier.
        $binDir = Split-Path -Parent $candidate
        $otherPaths = @($env:PATH -split ';' | Where-Object {
            $_ -and $_.TrimEnd('\') -ine $binDir.TrimEnd('\')
        })
        $env:PATH = (@($binDir) + $otherPaths) -join ';'
        Write-SubStep "CMake ${version}: $candidate"
        return $candidate
    }

    return $null
}

function Find-SevenZipExecutable {
    $candidates = @()
    $command = Get-Command 7z -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    foreach ($programFiles in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($programFiles) {
            $candidates += Join-Path $programFiles '7-Zip\7z.exe'
        }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        try {
            & $candidate -? *> $null
            if ($LASTEXITCODE -ne 0) { continue }
        } catch {
            continue
        }

        # An existing installation may not have added 7z.exe to PATH.
        if (-not $command -or $command.Source -ne $candidate) {
            $binDir = Split-Path -Parent $candidate
            $otherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $binDir })
            $env:PATH = (@($binDir) + $otherPaths) -join ';'
            Write-SubStep "Using 7-Zip from: $binDir"
        }
        return $candidate
    }
    return $null
}

function Find-MakeExecutable {
    $candidates = @()
    $command = Get-Command make -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    foreach ($programFiles in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
        if ($programFiles) {
            $candidates += Join-Path $programFiles 'GnuWin32\bin\make.exe'
        }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf) -or
            -not (Test-ExecutableVersion -Path $candidate)) { continue }

        if (-not $command -or $command.Source -ne $candidate) {
            $binDir = Split-Path -Parent $candidate
            $otherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $binDir })
            $env:PATH = (@($binDir) + $otherPaths) -join ';'
            Write-SubStep "Using GNU Make from: $binDir"
        }
        return $candidate
    }
    return $null
}

function Find-OfficialLlvmBin {
    $binDir = Join-Path $env:ProgramFiles 'LLVM\bin'
    $clangCl = Join-Path $binDir 'clang-cl.exe'
    $lldLink = Join-Path $binDir 'lld-link.exe'
    if ((Test-Path -LiteralPath $clangCl -PathType Leaf) -and
        (Test-Path -LiteralPath $lldLink -PathType Leaf) -and
        (Test-ExecutableVersion -Path $clangCl) -and
        (Test-ExecutableVersion -Path $lldLink)) {
        return $binDir
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
    $pathFixups = @(
        "$env:ProgramFiles\Git\usr\bin"                     # sh, bash, sed, grep (needed by tests)
        "${env:ProgramFiles(x86)}\GnuWin32\bin"             # make (GnuWin32 default)
        "$env:ProgramFiles\GnuWin32\bin"                    # make (alt location)
    )
    foreach ($dir in $pathFixups) {
        if ((Test-Path $dir) -and ($env:PATH -notlike "*$dir*")) {
            $env:PATH = "$env:PATH;$dir"
            Write-SubStep "Added to PATH: $dir"
        }
    }

    $cmakePath = Find-CMakeExecutable

    # CPack finds WiX through PATH. An older candle/light pair can appear
    # before the installed 3.14 tools (for example, in an Unreal SDK).
    $wixBin = Find-Wix314Bin
    if ($wixBin) {
        $otherPaths = @($env:PATH -split ';' | Where-Object { $_ -and $_ -ne $wixBin })
        $env:PATH = (@($wixBin) + $otherPaths) -join ';'
        Write-SubStep "Using WiX 3.14 from: $wixBin"
    }

    # Check all required tools.
    $required = @(
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

    $missing = @()
    foreach ($tool in $required) {
        $cmd = Get-Command $tool.Cmd -ErrorAction SilentlyContinue
        $toolPath = if ($cmd) { $cmd.Source } else { $null }
        # Windows may expose a python.exe App Execution Alias that is not an
        # installed interpreter.
        if ($cmd -and $tool.Cmd -eq 'python' -and
            -not (Test-ExecutableVersion -Path $cmd.Source)) {
            $toolPath = $null
        }
        if ($tool.Cmd -eq '7z') {
            $toolPath = Find-SevenZipExecutable
        }
        if ($tool.Cmd -eq 'make') {
            $toolPath = Find-MakeExecutable
        }
        # These tools are deprecated and removed in newer version of WiX.
        if ($tool.Cmd -in @('candle', 'light') -and -not $wixBin) {
            $toolPath = $null
        }
        if ($toolPath) {
            Write-SubStep "$($tool.Name): $toolPath"
        } else {
            $missing += $tool.Name
        }
    }

    if (-not $cmakePath) {
        $missing += "CMake $($script:MinimumCMakeVersion) or newer"
    }

    if (-not $ForceMSVC) {
        $llvmBin = Find-OfficialLlvmBin
        if ($llvmBin) {
            Write-SubStep "LLVM release (clang-cl and lld-link): $llvmBin"
        } else {
            $missing += 'LLVM release (clang-cl and lld-link)'
        }
    }

    # Lit disables the Python user site when running tests. Check the Python
    # installation itself so timeouts and tests requiring them stay available.
    $hostPython = Get-PythonExecutableForArch -Arch 'amd64'
    if ($hostPython) {
        if (Test-PythonPsutil -PythonExecutable $hostPython) {
            Write-SubStep "Python psutil: $hostPython"
        } else {
            $missing += 'Python psutil'
        }
    }
    if ($script:BuildArch -eq 'arm64') {
        $arm64Python = Get-PythonExecutableForArch -Arch 'arm64'
        if ($arm64Python -and $arm64Python -ne $hostPython) {
            if (Test-PythonPsutil -PythonExecutable $arm64Python) {
                Write-SubStep "Python psutil (ARM64): $arm64Python"
            } else {
                $missing += 'Python psutil (ARM64)'
            }
        }
    }

    if (Test-NetFx3Installed) {
        Write-SubStep 'NetFx3 (.NET Framework 3.5): installed'
    } else {
        $missing += 'NetFx3 (.NET Framework 3.5)'
    }

    return $missing
}

function Assert-SevenZipSymlinkSupport {
    # 7-Zip 21+ extracts symlinks from LLVM's source archive. Windows must
    # allow symlink creation when -DownloadSource is selected.
    $banner = (& 7z 2>&1 | Select-Object -First 2) -join ' '
    if ($banner -notmatch '7-Zip\s+(\d+)\.' -or [int]$Matches[1] -lt 21) {
        return
    }

    $linkPath = Join-Path ([IO.Path]::GetTempPath()) ("llvm-release-" + [guid]::NewGuid())
    try {
        New-Item -ItemType SymbolicLink -Path $linkPath -Target $PWD.Path -ErrorAction Stop | Out-Null
    } catch {
        throw "7-Zip 21+ needs permission to create symlinks when extracting the source archive. Run elevated, enable Developer Mode, or use 7-Zip 20.x."
    } finally {
        Remove-Item -LiteralPath $linkPath -Force -ErrorAction SilentlyContinue
    }
}

#===============================================================================
# Prerequisite installation (for fresh VMs)
#===============================================================================

function Test-IsElevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-NetFx3Installed {
    # The .NET Framework 3.5 installer records this value on both Windows
    # Server and desktop Windows. WiX 3.14 needs the feature to run.
    $key = 'HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP\v3.5'
    $framework = Get-ItemProperty -LiteralPath $key -ErrorAction SilentlyContinue
    return $null -ne $framework -and $framework.Install -eq 1
}

function Install-NetFx3 {
    if (Test-NetFx3Installed) {
        Write-SubStep 'NetFx3 (.NET Framework 3.5) is already installed.'
        return
    }

    Write-SubStep 'Installing NetFx3 (.NET Framework 3.5)...'
    $installationType = (Get-ItemProperty `
        -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').InstallationType
    if ($installationType -like 'Server*') {
        # Windows Server exposes NetFx3 as the NET-Framework-Core server role.
        # Use the built-in Windows PowerShell to load ServerManager reliably,
        # even when this release script is running under PowerShell 7.
        $windowsPowerShell = Join-Path $env:SystemRoot `
            'System32\WindowsPowerShell\v1.0\powershell.exe'
        $serverCommand = @'
$ErrorActionPreference = 'Stop'
Import-Module ServerManager
$feature = Get-WindowsFeature -Name NET-Framework-Core
if (-not $feature) { throw 'NET-Framework-Core is not available on this server.' }
if (-not $feature.Installed) {
    $result = Install-WindowsFeature -Name NET-Framework-Core
    if (-not $result.Success) {
        throw "Install-WindowsFeature failed: $($result.ExitCode)"
    }
    if ($result.RestartNeeded -eq 'Yes') { exit 3010 }
}
'@
        $encodedCommand = [Convert]::ToBase64String(
            [Text.Encoding]::Unicode.GetBytes($serverCommand))
        & $windowsPowerShell -NoProfile -NonInteractive -EncodedCommand $encodedCommand
        if ($LASTEXITCODE -eq 3010) {
            throw 'NetFx3 installation requires a restart. Restart Windows and rerun the release script.'
        }
        if ($LASTEXITCODE -ne 0) {
            throw "NetFx3 installation failed (exit code $LASTEXITCODE). If Windows cannot find the feature files, install NET-Framework-Core from matching Windows Server media and rerun the release script."
        }
    } else {
        $result = Enable-WindowsOptionalFeature -Online -FeatureName NetFx3 `
            -All -NoRestart -ErrorAction Stop
        if ($result.RestartNeeded) {
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
    $machinePath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
    $env:PATH = "$machinePath;$userPath"
}

function Invoke-ElevatedPrerequisiteInstallation {
    $releaseRoot = $script:ReleaseScriptRoot.Replace("'", "''")
    $commonPath = (Join-Path $script:ReleaseScriptRoot 'windows\Common.ps1').Replace("'", "''")
    $prerequisitesPath = (Join-Path $script:ReleaseScriptRoot 'windows\Prerequisites.ps1').Replace("'", "''")
    $toolchainPath = (Join-Path $script:ReleaseScriptRoot 'windows\Toolchain.ps1').Replace("'", "''")
    $buildArch = $script:BuildArch.Replace("'", "''")
    $minimumCMakeVersion = $script:MinimumCMakeVersion.ToString()
    $forceMSVCValue = if ($ForceMSVC) { '$true' } else { '$false' }
    $unattendedValue = if ($Unattended) { '$true' } else { '$false' }

    # Keep the elevated window open so the user can review the result.
    $pauseCommand = if (-not $Unattended) {
        "Read-Host 'Press Enter to close this window' | Out-Null"
    } else {
        ''
    }
    $workerCommand = @"
Set-StrictMode -Version Latest
`$ErrorActionPreference = 'Stop'
`$script:ReleaseScriptRoot = '$releaseRoot'
`$script:MinimumCMakeVersion = [version]'$minimumCMakeVersion'
`$script:BuildArch = '$buildArch'
`$ForceMSVC = $forceMSVCValue
`$Unattended = $unattendedValue
. '$commonPath'
. '$prerequisitesPath'
. '$toolchainPath'

`$exitCode = 0
try {
    Install-Prerequisites
} catch {
    Write-Host "Prerequisite installation failed:" -ForegroundColor Red
    Write-Host (`$_ | Out-String) -ForegroundColor Red
    `$exitCode = 1
}
$pauseCommand
exit `$exitCode
"@
    $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($workerCommand))

    Write-SubStep 'Requesting administrator access to install prerequisites...'
    try {
        $process = Start-Process -FilePath (Join-Path $PSHOME 'pwsh.exe') `
            -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encodedCommand) `
            -Verb RunAs -Wait -PassThru
    } catch {
        throw "Could not start elevated prerequisite installation: $_"
    }
    if ($process.ExitCode -ne 0) {
        throw "Elevated prerequisite installation failed (exit code $($process.ExitCode))."
    }
}

function Find-WingetExecutable {
    $candidates = @()
    $winget = Get-Command winget -CommandType Application -ErrorAction SilentlyContinue
    if ($winget) { $candidates += $winget.Source }
    if ($env:LOCALAPPDATA) {
        $candidates += Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\winget.exe'
    }

    # On fresh Windows Server images the App Installer alias may be absent
    # from the elevated PATH. Discover the installed package directly too.
    try {
        $packages = @(Get-AppxPackage -AllUsers -Name Microsoft.DesktopAppInstaller -ErrorAction Stop)
        foreach ($package in ($packages | Sort-Object Version -Descending)) {
            if ($package.InstallLocation) {
                $candidates += Join-Path $package.InstallLocation 'winget.exe'
            }
        }
    } catch {
        # Appx discovery is not available on every Windows image.
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        try {
            & $candidate --version *> $null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch {
            # Try the next location.
        }
    }
    throw "winget cannot run in the elevated shell. Checked: $($candidates -join ', '). Ensure App Installer is available to the account used by UAC."
}

function Assert-WixToolsUsable {
    $binDir = Find-Wix314Bin
    if (-not $binDir) {
        throw 'WiX 3.14 was installed, but candle.exe and light.exe could not run.'
    }
    Write-SubStep "WiX tools verified: $binDir"
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
    $wingetPath = Find-WingetExecutable

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
    $tools = @(
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

    foreach ($tool in $tools) {
        $cmd = Get-Command $tool.Verify -ErrorAction SilentlyContinue
        if ($cmd -and $tool.Verify -eq 'python' -and
            -not (Test-ExecutableVersion -Path $cmd.Source)) {
            $cmd = $null
        }
        $existingPath = if ($tool.Name -eq 'cmake') { Find-CMakeExecutable }
            elseif ($tool.Name -eq 'wix') { Find-Wix314Bin }
            elseif ($tool.Name -eq '7z') { Find-SevenZipExecutable }
            elseif ($tool.Name -eq 'make') { Find-MakeExecutable }
            elseif ($cmd) { $cmd.Source }
        if ($existingPath) {
            Write-SubStep "$($tool.Name) is already installed: $existingPath"
        } else {
            $versionArg = if ($tool.Version) { @("--version", $tool.Version) } else { @() }
            $versionLabel = if ($tool.Version) { " $($tool.Version)" } else { '' }
            Write-SubStep "Installing $($tool.Name) ($($tool.WingetId)$versionLabel)..."
            $installArgs = @('install', '--id', $tool.WingetId) + $versionArg +
                @('--accept-package-agreements', '--accept-source-agreements', '--silent')
            & $wingetPath @installArgs
            $installExitCode = $LASTEXITCODE
            if ($tool.Name -eq 'wix' -and $installExitCode -eq -1978334960) {
                # winget can fail while checking its NetFx3 dependency on
                # Server 2025 even after the feature is installed. The feature
                # was installed and verified above; bypass only this check.
                Write-Warning 'winget could not verify the installed NetFx3 feature. Retrying WiX with --force.'
                & $wingetPath @installArgs --force
                $installExitCode = $LASTEXITCODE
            }
            if ($tool.Name -eq '7z') {
                $sevenZipPath = Find-SevenZipExecutable
                if ($sevenZipPath) {
                    if ($installExitCode -ne 0) {
                        Write-Warning "winget exited with code $installExitCode, but 7-Zip is usable at $sevenZipPath. Continuing."
                    }
                    Write-SubStep "7-Zip verified: $sevenZipPath"
                    continue
                }
            }
            if ($tool.Name -eq 'make') {
                $makePath = Find-MakeExecutable
                if ($makePath) {
                    if ($installExitCode -ne 0) {
                        Write-Warning "winget exited with code $installExitCode, but GNU Make is usable at $makePath. Continuing."
                    }
                    Write-SubStep "GNU Make verified: $makePath"
                    continue
                }
            }
            if ($tool.Name -eq 'cmake') {
                if (-not (Find-CMakeExecutable)) {
                    throw "CMake installation finished, but version $($script:MinimumCMakeVersion) or newer could not be found."
                }
                continue
            }
            if ($installExitCode -ne 0) {
                if ($tool.Name -eq 'wix') {
                    throw "Failed to install WiX via winget (exit code $installExitCode). NetFx3 is installed; inspect winget's error above."
                }
                throw "Failed to install $($tool.Name) via winget (exit code $installExitCode)."
            }
            if ($tool.Name -eq '7z') {
                throw '7-Zip installation finished, but 7z.exe could not be found or run.'
            }
            if ($tool.Name -eq 'make') {
                throw 'GNU Make installation finished, but make.exe could not be found or run.'
            }
            if ($tool.Name -eq 'wix') { Assert-WixToolsUsable }
        }
    }

    if (-not $ForceMSVC) {
        $llvmBin = Find-OfficialLlvmBin
        if ($llvmBin) {
            Write-SubStep "LLVM release is already installed: $llvmBin"
        } else {
            Write-SubStep 'Installing LLVM release (LLVM.LLVM)...'
            & $wingetPath install --id LLVM.LLVM --exact --source winget `
                --accept-package-agreements --accept-source-agreements --silent
            $installExitCode = $LASTEXITCODE
            $llvmBin = Find-OfficialLlvmBin
            if (-not $llvmBin) {
                throw "LLVM installation did not provide usable clang-cl.exe and lld-link.exe in $env:ProgramFiles\LLVM\bin (winget exit code $installExitCode)."
            }
            if ($installExitCode -ne 0) {
                Write-Warning "winget exited with code $installExitCode, but LLVM tools are usable at $llvmBin. Continuing."
            }
            Write-SubStep "LLVM release verified: $llvmBin"
        }
    }

    # Install ARM64 Python when building ARM64 so LLDB uses a matching Python.
    # The main Python install above covers the runner's default architecture.
    $archPython = @()
    if ($script:BuildArch -eq 'arm64') { $archPython += @{ Arch = 'arm64'; Suffix = '-arm64' } }
    foreach ($ap in $archPython) {
        $basePath = "$env:LOCALAPPDATA\Programs\Python"
        $existing = if (Test-Path $basePath) {
            Get-ChildItem -Path $basePath -Directory -Filter "Python3*$($ap.Suffix)" |
                Where-Object { Test-Path (Join-Path $_.FullName 'python.exe') }
        } else { $null }
        if ($existing) {
            Write-SubStep "Python for $($ap.Arch) already installed: $($existing[0].FullName)"
        } else {
            Write-SubStep "Installing Python for $($ap.Arch) (Python.Python.3.13 --architecture $($ap.Arch))..."
            & $wingetPath install --id Python.Python.3.13 --architecture $($ap.Arch) --accept-package-agreements --accept-source-agreements --silent
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Failed to install Python for $($ap.Arch). LLDB may not work for this architecture."
            }
        }
    }

    # Visual Studio Build Tools - special handling
    Install-VisualStudio -WingetPath $wingetPath

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

    $requiredComponents = @(
        "Microsoft.VisualStudio.Workload.VCTools"
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
        "Microsoft.VisualStudio.Component.VC.Tools.ARM64"
        "Microsoft.VisualStudio.Component.VC.ATL"
        "Microsoft.VisualStudio.Component.VC.ATL.ARM64"
        "Microsoft.VisualStudio.Component.VC.DiagnosticTools"  # includes DIA SDK
        # The latest Windows SDK is included automatically via --includeRecommended
    )

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -nologo -latest -products '*' -format json | ConvertFrom-Json
        if ($vs) {
            Write-SubStep "Visual Studio already installed: $($vs[0].installationPath)"
            Write-SubStep "Ensuring required components are installed..."
            # setup.exe modify does not accept --wait; wait for the process
            # from PowerShell instead.
            $addArgs = @()
            foreach ($component in $requiredComponents) {
                $addArgs += '--add'
                $addArgs += $component
            }
            $installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\setup.exe"
            if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
                throw "Visual Studio Installer not found: $installer"
            }
            $installPath = $vs[0].installationPath
            $installArgs = @('modify', '--installPath', "`"$installPath`"") +
                $addArgs + @('--includeRecommended', '--passive', '--norestart')
            $process = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
            if ($process.ExitCode -eq 3010) {
                throw 'Visual Studio components were installed, but Windows must be restarted before building.'
            }
            if ($process.ExitCode -ne 0) {
                throw "Failed to modify Visual Studio installation (exit code $($process.ExitCode))."
            }
            return
        }
    }

    Write-SubStep "Installing Visual Studio Build Tools 2022..."
    $addArgs = ($requiredComponents | ForEach-Object { "--add $_" }) -join ' '
    & $WingetPath install Microsoft.VisualStudio.2022.BuildTools `
        --accept-package-agreements --accept-source-agreements --silent `
        --override "$addArgs --includeRecommended --passive --wait"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Visual Studio Build Tools."
    }
}

