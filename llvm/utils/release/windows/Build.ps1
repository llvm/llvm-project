# PGO and architecture build pipeline.

function New-PGOProfile {
    <#
    .SYNOPSIS
        Builds and trains an instrumented stage 2 clang, and merges the
        profile into $ProfilePath.
    .NOTES
        PGO artifacts are created as top-level siblings of the stage 2 build
        directory so that each step owns an independent directory tree:
          $InstrumentDir/   - instrumented stage 2 clang build
          $TrainDir/        - training build
          $ProfilePath      - merged profile data
    #>
    param(
        [string[]]$CMakeFlags,
        [string]$Stage1BinDir,
        [string]$LLVMSrc,
        [Parameter(Mandatory)][string]$InstrumentDir,
        [Parameter(Mandatory)][string]$TrainDir,
        [Parameter(Mandatory)][string]$ProfilePath
    )
    Write-Step "Generating PGO profile for stage 2 self-hosted compiler"

    # Build instrumented stage 2 Clang
    Write-SubStep "Building instrumented stage 2 Clang..."
    New-Item -ItemType Directory -Path $InstrumentDir -Force | Out-Null
    Push-Location $InstrumentDir
    try {
        # The instrumented build only needs to produce an instrumented clang;
        # strip runtimes/projects that are unnecessary and would themselves be
        # compiled with -fprofile-generate (requiring the profile runtime to link).
        $InstrumentFlags = $CMakeFlags + @(
            "LLVM_TARGETS_TO_BUILD=Native"
            "LLVM_BUILD_INSTRUMENTED=IR"
            'LLVM_ENABLE_RUNTIMES=""'
            'LLVM_ENABLE_PROJECTS="clang;lld"'
        )
        $CacheFile = Write-CMakeCacheFile -Flags $InstrumentFlags -FileName 'instrument_cache.cmake'
        Invoke-NativeCommand cmake -G Ninja -C $CacheFile "$LLVMSrc/llvm"
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs clang
        $InstrumentedClang = ConvertTo-UnixPath (Join-Path $PWD 'bin' 'clang-cl.exe')
    } finally {
        Pop-Location
    }

    # Train: build part of LLVM with the instrumented stage 2 compiler
    Write-SubStep "Training with instrumented stage 2 Clang..."
    New-Item -ItemType Directory -Path $TrainDir -Force | Out-Null
    Push-Location $TrainDir
    try {
        $TrainFlags = $CMakeFlags + @(
            "CMAKE_C_COMPILER=$InstrumentedClang"
            "CMAKE_CXX_COMPILER=$InstrumentedClang"
            "LLVM_ENABLE_PROJECTS=clang"
            'LLVM_ENABLE_RUNTIMES=""'
            "LLVM_TARGETS_TO_BUILD=Native"
        )
        $CacheFile = Write-CMakeCacheFile -Flags $TrainFlags -FileName 'train_cache.cmake'
        Invoke-NativeCommand cmake -G Ninja -C $CacheFile "$LLVMSrc/llvm"

        # Drop profiles generated from running cmake; those are not representative.
        Remove-Item -Path "$InstrumentDir/profiles/*.profraw" -Force -ErrorAction SilentlyContinue
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs 'tools/clang/lib/Sema/CMakeFiles/obj.clangSema.dir/Sema.cpp.obj'
    } finally {
        Pop-Location
    }

    # Merge profiles
    $ResolvedProfilePath = ConvertTo-UnixPath $ProfilePath
    Invoke-NativeCommand "$Stage1BinDir/llvm-profdata" merge `
        -output="$ResolvedProfilePath" "$InstrumentDir/profiles/*.profraw"

    Write-SubStep "PGO profile generated: $ResolvedProfilePath"
}

function Get-NinjaInputs {
    <#
    .SYNOPSIS
        Lists the inputs of the build edge that produces a Ninja target.
    #>
    param([Parameter(Mandatory)][string]$Target)
    $InInputs = $false
    foreach ($Line in (Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs -t query $Target)) {
        # 'ninja -t query' lists the inputs under "  input: <rule>", with
        # "| " and "|| " before implicit and order-only inputs.
        if ($Line -match '^  input: ') {
            $InInputs = $true
        } elseif ($Line -match '^  \S') {
            $InInputs = $false
        } elseif ($InInputs -and $Line -match '^    (?:\|\|? )?(.+)$') {
            $Matches[1]
        }
    }
}

function Get-TestDependencyTargets {
    <#
    .SYNOPSIS
        Lists the targets that check-* targets build before they run lit.
    .NOTES
        CMake's Ninja generator writes check-clang as a phony alias for
        tools/clang/test/check-clang. That phony target's inputs are the lit
        command (tools/clang/test/CMakeFiles/check-clang) and the targets the
        tests depend on.
    #>
    param([Parameter(Mandatory)][string[]]$Targets)
    $Dependencies = [System.Collections.Generic.List[string]]::new()
    foreach ($Target in $Targets) {
        $Name = [regex]::Escape($Target)
        $LitCommand = "(^|[\\/])CMakeFiles[\\/]$Name$"
        $Inputs = @(Get-NinjaInputs $Target)
        while ($Inputs.Count -eq 1 -and $Inputs[0] -match "[\\/]$Name$" -and
               $Inputs[0] -notmatch $LitCommand) {
            $Inputs = @(Get-NinjaInputs $Inputs[0])
        }
        foreach ($Dependency in $Inputs) {
            if ($Dependency -notmatch $LitCommand -and -not $Dependencies.Contains($Dependency)) {
                $Dependencies.Add($Dependency)
            }
        }
        # check-runtimes builds the runtimes test dependencies in the same
        # runtimes build command that runs lit.
        if ($Target -eq 'check-runtimes' -and -not $Dependencies.Contains('runtimes-test-depends')) {
            $Dependencies.Add('runtimes-test-depends')
        }
    }
    return $Dependencies
}

function Invoke-Tests {
    <#
    .SYNOPSIS
        Builds the dependencies of a list of test targets, then runs each
        test target once.
        Honors $script:NinjaCommand and $script:NinjaExtraArgs (set via
        LLVM_NINJA_OVERRIDE).
    .NOTES
        A test dependency that fails to build stops the build. A failing test
        target does not: the failure is recorded and the remaining targets
        still run, so packaging can complete. The main script reports the
        recorded failures once the build finishes.
    #>
    param(
        [string[]]$Targets
    )
    $Dependencies = @(Get-TestDependencyTargets -Targets $Targets)
    if ($Dependencies.Count -gt 0) {
        Write-SubStep "Building test dependencies for: $($Targets -join ', ')"
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs @Dependencies
    }
    foreach ($Target in $Targets) {
        try {
            Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs $Target
        } catch {
            Write-Warning "Test target '$Target' failed; continuing. $($_.Exception.Message)"
            $script:FailedTestTargets.Add([pscustomobject]@{
                Target         = $Target
                BuildDirectory = $PWD.Path
            })
        }
    }
}

function Write-TestFailureSummary {
    <#
    .SYNOPSIS
        Reports the test targets that failed during the build. Under GitHub
        Actions, also emits a warning annotation for each failed target.
    #>
    if ($script:FailedTestTargets.Count -eq 0) {
        return
    }
    Write-Host ''
    Write-Warning ("$($script:FailedTestTargets.Count) test target(s) failed. " +
        'The packages were still created; review the lit output above before publishing them.')
    foreach ($Failure in $script:FailedTestTargets) {
        Write-Host "  $($Failure.Target) in $($Failure.BuildDirectory)" -ForegroundColor Yellow
        if ($env:GITHUB_ACTIONS -eq 'true') {
            Write-Host "::warning title=Windows release tests::$($Failure.Target) failed in $($Failure.BuildDirectory)"
        }
    }
}

#===============================================================================
# Main build stages
#===============================================================================

function Build-Stage {
    <#
    .SYNOPSIS
        Builds the stage 1 bootstrap or stage 2 self-hosted compiler.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [Parameter(Mandatory)]
        [string]$Description,
        [Parameter(Mandatory)]
        [string[]]$CMakeFlags,
        [Parameter(Mandatory)]
        [string]$LLVMSrc,
        [Parameter(Mandatory)]
        [string]$StaticCache,
        [string[]]$ExtraCMakeFlags = @(),
        [string[]]$InitialBuildTargets = @(),
        [switch]$SkipFullBuild,
        [string[]]$TestTargets = @()
    )

    Write-Step "Building $Description ($Name)"
    New-Item -ItemType Directory -Path $Name -Force | Out-Null
    Push-Location $Name
    try {
        $AllFlags = $CMakeFlags + $ExtraCMakeFlags
        $CacheFile = Write-CMakeCacheFile -Flags $AllFlags
        # The static cache is last so its fixed settings override the shared flags.
        Invoke-NativeCommand cmake -G Ninja -C $CacheFile -C $StaticCache "$LLVMSrc/llvm"
        if ($InitialBuildTargets.Count -gt 0) {
            Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs @InitialBuildTargets
        }
        if (-not $SkipFullBuild) {
            Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs
            if ($TestTargets.Count -gt 0) {
                Invoke-Tests -Targets $TestTargets
            }
        }
    } finally {
        Pop-Location
    }
}


function Build-Architecture {
    <#
    .SYNOPSIS
        Selects the stage 0 host toolchain, then builds the stage 1 bootstrap
        and stage 2 self-hosted compilers for an architecture.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Arch,
        [Parameter(Mandatory)]
        [string]$VsDevCmd,
        [Parameter(Mandatory)]
        [string]$LLVMSrc,
        [Parameter(Mandatory)]
        [string]$BuildDir,
        [Parameter(Mandatory)]
        [string]$ThirdPartyDir,
        [Parameter(Mandatory)]
        [string]$PackageVersion,
        [Parameter(Mandatory)]
        [string[]]$CommonCMakeFlags,
        [Parameter(Mandatory)]
        [string]$CommonCompilerFlags,
        [Parameter(Mandatory)]
        [string[]]$Stage2CMakeFlags,
        [switch]$UseFastBuild
    )

    Write-Step "Building for architecture: $Arch"

    # Restore clean PATH and setup environment
    $env:PATH = $script:OriginalPath
    $PythonHome = Find-Python -Arch $Arch
    $PythonExecutable = Join-Path $PythonHome 'python.exe'
    if (-not (Test-PythonPsutil -PythonExecutable $PythonExecutable)) {
        throw "Python psutil is missing from $PythonExecutable. Run with -InstallPrerequisites to enable lit test timeouts."
    }
    Enter-VsDevEnvironment -VsDevCmd $VsDevCmd -Arch $Arch
    Use-DiaSdkRuntime -Arch $Arch

    # Stage 0 uses the official LLVM release unless MSVC was requested.
    $HostClangCl = $null
    $HostLldLink = $null
    if (-not $ForceMSVC) {
        $LLVMBin = Find-OfficialLlvmBin
        if (-not $LLVMBin) {
            throw 'Official LLVM clang-cl.exe and lld-link.exe are missing. Run with -InstallPrerequisites, or use -ForceMSVC.'
        }
        $env:PATH = "$LLVMBin;$env:PATH"
        $HostClangCl = Join-Path $LLVMBin 'clang-cl.exe'
        $HostLldLink = Join-Path $LLVMBin 'lld-link.exe'
    }

    if ($HostClangCl) {
        $HostCompiler = $HostClangCl
        $HostLinker = $HostLldLink
        $CommonCompilerFlags += ' -fuse-ld=lld'
        $CommonCMakeFlags += @(
            "LLVM_ENABLE_LLD=ON"
            "CMAKE_C_FLAGS=`"$CommonCompilerFlags`""
            "CMAKE_CXX_FLAGS=`"$CommonCompilerFlags`""
        )
    } else {
        # Pin the MSVC driver so bcrypt.lib is interpreted as a library.
        $HostCompiler = 'cl.exe'
        $HostLinker = 'link.exe'
    }
    Write-SubStep "Stage 0 host compiler: $HostCompiler (linker: $HostLinker)"

    $HostToolchainFlags = @(
        "CMAKE_C_COMPILER=$HostCompiler"
        "CMAKE_CXX_COMPILER=$HostCompiler"
        "CMAKE_LINKER=$HostLinker"
    )
    $CommonCMakeFlags += $HostToolchainFlags

    $env:VSCMD_START_DIR = $BuildDir
    $SwigConfig = Find-Swig
    # CMake and Ninja subprocesses need the library when invoking SWIG.
    $env:SWIG_LIB = $SwigConfig.LibraryDirectory

    $ReleaseWindowsDir = Join-Path $script:ReleaseScriptRoot 'windows'
    $Stage1Cache = Join-Path $ReleaseWindowsDir 'stage1-cache.cmake'
    $Stage2Cache = Join-Path $ReleaseWindowsDir 'stage2-cache.cmake'

    # Directory names (always computed, used by multiple steps)
    $Stage1Name = "build_${Arch}_stage1"
    $Stage2Name = "build_${Arch}_stage2"
    $InstrumentDir = Join-Path $BuildDir "instrument_${Arch}_stage2"
    $TrainDir      = Join-Path $BuildDir "train_${Arch}_stage2"
    $ProfileFile   = Join-Path $BuildDir "profile_${Arch}_stage2.profdata"

    #-------------------------------------------------------------------
    # Step: dependencies (libxml2, zlib, and zstd)
    # Each dependency returns immediately when this step is skipped.
    #-------------------------------------------------------------------
    foreach ($Library in $script:ThirdPartyLibraries) {
        Build-ThirdPartyLibrary -Library $Library `
            -SourceDir (Join-Path $ThirdPartyDir $Library.SourceDirectory) `
            -BuildDirectory (Join-Path $BuildDir $Stage1Name $Library.BuildDirectory) `
            -ToolchainFlags $HostToolchainFlags
    }
    $LibXmlDir = $script:ThirdPartyInstallDirs['libxml2']
    $ZlibDir = $script:ThirdPartyInstallDirs['zlib']
    $ZstdDir = $script:ThirdPartyInstallDirs['zstd']

    # The stage 1 bootstrap tools are needed by PGO and stage 2.
    $Stage1BinDir = ConvertTo-UnixPath (Join-Path $BuildDir $Stage1Name 'bin')

    # Compute cmakeFlags (always needed, cheap -- derived from parameters)
    $CMakeFlags = $CommonCMakeFlags + @(
        "Python3_ROOT_DIR=$env:PYTHONHOME"
        "LIBXML2_INCLUDE_DIR=$LibXmlDir/include/libxml2"
        "LIBXML2_LIBRARY=$LibXmlDir/lib/libxml2s.lib"
        "LIBXML2_LIBRARIES=$LibXmlDir/lib/libxml2s.lib;ws2_32.lib"
        "ZLIB_INCLUDE_DIR=$ZlibDir/include"
        "ZLIB_LIBRARY=$ZlibDir/lib/zs.lib"
        "ZLIB_LIBRARY_RELEASE=$ZlibDir/lib/zs.lib"
        "zstd_INCLUDE_DIR=$ZstdDir/include"
        "zstd_LIBRARY=$ZstdDir/lib/zstd_static.lib"
    )

    $CMakeFlags += "CLANG_DEFAULT_LINKER=lld"
    if ($Arch -eq 'arm64') {
        $CMakeFlags += "COMPILER_RT_BUILD_SANITIZERS=OFF"
    }

    #-------------------------------------------------------------------
    # Step: stage 1 bootstrap compiler
    #-------------------------------------------------------------------
    if (Test-ShouldRun 'stage1') {
        # Preserve the dependency installs needed by stage 1.
        $Stage1Dir = Join-Path $BuildDir $Stage1Name
        if (Test-Path $Stage1Dir) {
            Write-SubStep "Cleaning stage 1 bootstrap build artifacts (preserving dependency installs)..."
            $PreservedBuilds = @($script:ThirdPartyLibraries | ForEach-Object { $_.BuildDirectory })
            Get-ChildItem -Path $Stage1Dir -Exclude $PreservedBuilds |
                ForEach-Object {
                    Remove-ItemWithoutProgress -LiteralPath $_.FullName -Recurse -Force
                }
        }

        # Stage 2 needs these tools even when the fast build skips the rest
        # of the stage 1 bootstrap build and its tests.
        $Stage1InitialTargets = @('clang', 'lld', 'llvm-lib', 'llvm-rc', 'runtimes')
        if ($Arch -eq 'amd64') {
            $Stage1InitialTargets += 'llvm-ml64'
        }
        $SkipStage1FullBuild = $UseFastBuild
        $Stage1Tests = @('check-llvm', 'check-clang', 'check-lld', 'check-runtimes')

        Build-Stage -Name $Stage1Name -Description 'stage 1 bootstrap compiler' `
            -CMakeFlags $CMakeFlags `
            -LLVMSrc $LLVMSrc `
            -StaticCache $Stage1Cache `
            -InitialBuildTargets $Stage1InitialTargets `
            -SkipFullBuild:$SkipStage1FullBuild `
            -TestTargets $Stage1Tests
    } else {
        Write-Step "Skipping stage 1 bootstrap compiler (-StartAt $($script:StartAtStep))"
        Assert-PathExists -Path "$Stage1BinDir/clang-cl.exe" -Description 'stage 1 bootstrap clang-cl.exe'
        Assert-PathExists -Path "$Stage1BinDir/lld-link.exe" -Description 'stage 1 bootstrap lld-link.exe'
        Assert-PathExists -Path "$Stage1BinDir/llvm-lib.exe" -Description 'stage 1 bootstrap llvm-lib.exe'
        Assert-PathExists -Path "$Stage1BinDir/llvm-rc.exe" -Description 'stage 1 bootstrap llvm-rc.exe'
        if ($Arch -eq 'amd64') {
            Assert-PathExists -Path "$Stage1BinDir/llvm-ml64.exe" -Description 'stage 1 bootstrap llvm-ml64.exe'
        }
        # -FastBuild does not build llvm-profdata, which the PGO merge needs.
        if (-not $UseFastBuild -and (Test-ShouldRun 'pgo')) {
            Assert-PathExists -Path "$Stage1BinDir/llvm-profdata.exe" -Description 'stage 1 bootstrap llvm-profdata.exe (not built by -FastBuild)'
        }
    }

    # Stage 2 is built with the stage 1 bootstrap toolchain.
    $Stage2CompilerFlags = @(
        "CMAKE_C_COMPILER=$Stage1BinDir/clang-cl.exe"
        "CMAKE_CXX_COMPILER=$Stage1BinDir/clang-cl.exe"
        "CMAKE_LINKER=$Stage1BinDir/lld-link.exe"
        "CMAKE_AR=$Stage1BinDir/llvm-lib.exe"
        "CMAKE_RC_COMPILER=$Stage1BinDir/llvm-rc.exe"
    )

    if ($Arch -eq 'amd64') {
        $Stage2CompilerFlags += "CMAKE_ASM_MASM_COMPILER=$Stage1BinDir/llvm-ml64.exe"
    } else {
        $Stage2CompilerFlags += "CPACK_SYSTEM_NAME=woa64"
    }

    $Stage2Flags = $CMakeFlags + $Stage2CompilerFlags

    #-------------------------------------------------------------------
    # Step: pgo
    #-------------------------------------------------------------------
    $ProfileFlags = @()
    if (-not $UseFastBuild) {
        if (Test-ShouldRun 'pgo') {
            Remove-StepDirectory $InstrumentDir
            Remove-StepDirectory $TrainDir
            Remove-StepDirectory $ProfileFile
            New-PGOProfile -CMakeFlags $Stage2Flags `
                -Stage1BinDir $Stage1BinDir -LLVMSrc $LLVMSrc `
                -InstrumentDir $InstrumentDir `
                -TrainDir $TrainDir `
                -ProfilePath $ProfileFile
        } else {
            Write-Step "Skipping PGO (-StartAt $($script:StartAtStep))"
            Assert-PathExists -Path $ProfileFile -Description 'PGO profile data'
        }
        $ProfilePath = ConvertTo-UnixPath $ProfileFile

        $PgoCFlags = "$CommonCompilerFlags -Wno-backend-plugin"
        $ProfileFlags = @(
            "LLVM_PROFDATA_FILE=$ProfilePath"
            "CMAKE_C_FLAGS=`"$PgoCFlags`""
            "CMAKE_CXX_FLAGS=`"$PgoCFlags`""
        )
    }

    #-------------------------------------------------------------------
    # Step: stage 2 self-hosted compiler
    # Fixed project, runtime, and LLDB settings are in stage2-cache.cmake.
    #-------------------------------------------------------------------
    $Stage2Extra = @(
        "PYTHON_HOME=$env:PYTHONHOME"
        # The tarball step overrides these settings. Set them explicitly so
        # that -StartAt package restores them. CPack installs the MSI under
        # CPACK_PACKAGE_INSTALL_DIRECTORY, so the prefix does not affect it.
        "CMAKE_INSTALL_PREFIX:PATH=$env:ProgramFiles\LLVM"
        "LLVM_INCLUDE_TESTS:BOOL=ON"
    ) + $Stage2CMakeFlags + @(
        "SWIG_EXECUTABLE=$($SwigConfig.Executable)"
        "SWIG_DIR=$($SwigConfig.LibraryDirectory)"
    ) + $ProfileFlags

    if (Test-ShouldRun 'stage2') {
        # Also discards the settings the tarball step left in the CMake cache.
        Remove-StepDirectory (Join-Path $BuildDir $Stage2Name)

        # check-clang-tools also runs the clangd tests.
        $Stage2Tests = @('check-llvm', 'check-clang', 'check-lld', 'check-runtimes',
                         'check-clang-tools')

        Build-Stage -Name $Stage2Name -Description 'stage 2 self-hosted compiler' `
            -CMakeFlags $Stage2Flags `
            -LLVMSrc $LLVMSrc `
            -StaticCache $Stage2Cache `
            -ExtraCMakeFlags $Stage2Extra `
            -TestTargets $Stage2Tests
    } else {
        Write-Step "Skipping stage 2 self-hosted compiler (-StartAt $($script:StartAtStep))"
        Assert-PathExists -Path (Join-Path $BuildDir $Stage2Name) -Description 'stage 2 build directory'
    }

    $TripleArch = if ($Arch -eq 'amd64') { 'x86_64' } else { 'aarch64' }
    $Filename = "clang+llvm-${PackageVersion}-${TripleArch}-pc-windows-msvc"
    $TarballInstallDirectory = Join-Path $BuildDir $Filename

    #-------------------------------------------------------------------
    # Step: package (WiX MSI installer)
    #-------------------------------------------------------------------
    if (Test-ShouldRun 'package') {
        Write-Step "Creating WiX MSI installer package"
        Push-Location $Stage2Name
        try {
            $TarballConfigMarker = Join-Path $PWD '.llvm_release_tarball_configured'
            if (Test-TarballConfigurationApplied -BuildDirectory $PWD.Path `
                    -TarballInstallDirectory $TarballInstallDirectory) {
                Write-SubStep 'The stage2 cache has tarball settings; restoring the MSI build configuration.'
                $PackageFlags = $Stage2Flags + $Stage2Extra
                $CacheFile = Write-CMakeCacheFile -Flags $PackageFlags -FileName 'package_cache.cmake'
                Invoke-NativeCommand cmake -G Ninja -C $CacheFile -C $Stage2Cache "$LLVMSrc/llvm"
                Remove-Item -LiteralPath $TarballConfigMarker -Force -ErrorAction SilentlyContinue
            } else {
                Write-SubStep 'Stage2 is already configured for packaging; skipping CMake reconfigure.'
            }

            # CPack does not remove stale files from its staging tree, so
            # clean it in case a previous package attempt left staged files.
            Remove-StepDirectory (Join-Path $PWD '_CPack_Packages')

            Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs package
            # Downloaded release tags can predate these MSI settings. Check
            # each invariant only when the source being packaged defines it.
            $LLVMCMakeLists = Join-Path $LLVMSrc 'llvm/CMakeLists.txt'
            if (Select-String -LiteralPath $LLVMCMakeLists -Pattern 'set\(CPACK_WIX_UPGRADE_GUID' -Quiet) {
                Assert-MsiUpgradeCode -BuildDirectory $PWD.Path
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Step "Skipping package (-StartAt $($script:StartAtStep))"
    }

    #-------------------------------------------------------------------
    # Step: tarball
    #-------------------------------------------------------------------
    if (Test-ShouldRun 'tarball') {
        Remove-StepDirectory (Join-Path $BuildDir $Filename)
        Remove-StepDirectory (Join-Path $BuildDir "$Filename.tar")
        Remove-StepDirectory (Join-Path $BuildDir "$Filename.tar.xz")

        Push-Location $Stage2Name
        try {
            $TarballFlags = $Stage2Flags + $Stage2Extra + @(
                "LLVM_INSTALL_TOOLCHAIN_ONLY=OFF"
                "CMAKE_INSTALL_PREFIX=$BuildDir/$Filename"
                "LLVM_INCLUDE_TESTS=OFF"
            )
            $CacheFile = Write-CMakeCacheFile -Flags $TarballFlags -FileName 'tarball_cache.cmake'
            # Leave a durable marker so a later -StartAt package invocation
            # restores the stage2 settings before invoking CPack.
            Set-Content -LiteralPath '.llvm_release_tarball_configured' `
                -Value 'LLVM_INSTALL_TOOLCHAIN_ONLY=OFF' -Encoding ASCII
            Invoke-NativeCommand cmake -G Ninja -C $CacheFile -C $Stage2Cache "$LLVMSrc/llvm"
            Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs install

            # Verify llvm-config works
            Invoke-NativeCommand "$BuildDir/$Filename/bin/llvm-config.exe" --bindir
        } finally {
            Pop-Location
        }

        # Create the portable archive without changing executable layout.
        Invoke-NativeCommand 7z a -ttar "$Filename.tar" $Filename
        Invoke-NativeCommand 7z a -txz "$Filename.tar.xz" "$Filename.tar"
        Remove-Item -LiteralPath "$Filename.tar"
    } else {
        Write-Step "Skipping tarball (-StartAt $($script:StartAtStep))"
    }
}
