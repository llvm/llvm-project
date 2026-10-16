# PGO and architecture build pipeline.

function New-PGOProfile {
    <#
    .SYNOPSIS
        Builds and trains an instrumented stage 2 clang and sets the profile path.
    .NOTES
        Uses a script-scoped variable instead of return to avoid PowerShell
        capturing native command stdout as part of the function's return value,
        which would break Ninja's \r progress display and corrupt the path.

        PGO artifacts are created as top-level siblings of the stage 2 build
        directory so that each step owns an independent directory tree:
          $InstrumentDir/   - instrumented stage 2 clang build
          $TrainDir/        - training build
          $ProfilePath      - merged profile data
    #>
    param(
        [string[]]$CMakeFlags,
        [string]$Stage1BinDir,
        [string]$LlvmSrc,
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
        $instrumentFlags = $CMakeFlags + @(
            "LLVM_TARGETS_TO_BUILD=Native"
            "LLVM_BUILD_INSTRUMENTED=IR"
            'LLVM_ENABLE_RUNTIMES=""'
            'LLVM_ENABLE_PROJECTS="clang;lld"'
        )
        $cache = Write-CMakeCacheFile -Flags $instrumentFlags -FileName 'instrument_cache.cmake'
        $otherFlags = $cache.OtherFlags
        Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$LlvmSrc/llvm"
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs clang
        $instrumentedClang = ConvertTo-ForwardSlashPath (Join-Path $PWD 'bin' 'clang-cl.exe')
    } finally {
        Pop-Location
    }

    # Train: build part of LLVM with the instrumented stage 2 compiler
    Write-SubStep "Training with instrumented stage 2 Clang..."
    New-Item -ItemType Directory -Path $TrainDir -Force | Out-Null
    Push-Location $TrainDir
    try {
        $trainFlags = $CMakeFlags + @(
            "CMAKE_C_COMPILER=$instrumentedClang"
            "CMAKE_CXX_COMPILER=$instrumentedClang"
            "LLVM_ENABLE_PROJECTS=clang"
            'LLVM_ENABLE_RUNTIMES=""'
            "LLVM_TARGETS_TO_BUILD=Native"
        )
        $cache = Write-CMakeCacheFile -Flags $trainFlags -FileName 'train_cache.cmake'
        $otherFlags = $cache.OtherFlags
        Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$LlvmSrc/llvm"

        # Drop profiles generated from running cmake; those are not representative.
        Remove-Item -Path "$InstrumentDir/profiles/*.profraw" -Force -ErrorAction SilentlyContinue
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs 'tools/clang/lib/Sema/CMakeFiles/obj.clangSema.dir/Sema.cpp.obj'
    } finally {
        Pop-Location
    }

    # Merge profiles
    $resolvedProfilePath = ConvertTo-ForwardSlashPath $ProfilePath
    Invoke-NativeCommand "$Stage1BinDir/llvm-profdata" merge `
        -output="$resolvedProfilePath" "$InstrumentDir/profiles/*.profraw"

    Write-SubStep "PGO profile generated: $resolvedProfilePath"
    $script:PGOProfilePath = $resolvedProfilePath
}

function Invoke-Build {
    param([string[]]$Targets = @())
    Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs @Targets
}

function Invoke-Tests {
    <#
    .SYNOPSIS
        Runs a list of test targets once each, skipping as appropriate.
        Honors $script:NinjaCommand and $script:NinjaExtraArgs (set via
        LLVM_NINJA_OVERRIDE).
    #>
    param(
        [string[]]$Targets,
        [string]$Arch = ''
    )
    foreach ($target in $Targets) {
        # Skip runtime checks on non-amd64
        if ($target -eq 'check-runtimes' -and $Arch -ne 'amd64') {
            Write-SubStep "Skipping $target on $Arch"
            continue
        }
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs $target
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
        [string]$LlvmSrc,
        [string[]]$ExtraCMakeFlags = @(),
        [string[]]$InitialBuildTargets = @(),
        [switch]$SkipFullBuild,
        [string[]]$TestTargets = @(),
        [string]$Arch = ''
    )

    Write-Step "Building $Description ($Name)"
    New-Item -ItemType Directory -Path $Name -Force | Out-Null
    Push-Location $Name
    try {
        $allFlags = $CMakeFlags + $ExtraCMakeFlags
        $cache = Write-CMakeCacheFile -Flags $allFlags
        $otherFlags = $cache.OtherFlags
        Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$LlvmSrc/llvm"
        if ($InitialBuildTargets.Count -gt 0) {
            Invoke-Build -Targets $InitialBuildTargets
        }
        if (-not $SkipFullBuild) {
            Invoke-Build
            if ($TestTargets.Count -gt 0) {
                Invoke-Tests -Targets $TestTargets -Arch $Arch
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
        [string]$LlvmSrc,
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
        [string[]]$CommonLLDBFlags = @(),
        [switch]$UseFastBuild
    )

    Write-Step "Building for architecture: $Arch"

    # Restore clean PATH and setup environment
    $env:PATH = $script:OriginalPath
    $pythonHome = Find-Python -Arch $Arch
    $pythonExecutable = Join-Path $pythonHome 'python.exe'
    if (-not (Test-PythonPsutil -PythonExecutable $pythonExecutable)) {
        throw "Python psutil is missing from $pythonExecutable. Run with -InstallPrerequisites to enable lit test timeouts."
    }
    Enter-VsDevEnvironment -VsDevCmd $VsDevCmd -Arch $Arch
    Use-DiaSdkRuntime -Arch $Arch

    # Stage 0 uses the official LLVM release unless MSVC was requested.
    $hostClangCl = $null
    $hostLldLink = $null
    if (-not $ForceMSVC) {
        $llvmBin = Find-OfficialLlvmBin
        if (-not $llvmBin) {
            throw 'Official LLVM clang-cl.exe and lld-link.exe are missing. Run with -InstallPrerequisites, or use -ForceMSVC.'
        }
        $env:PATH = "$llvmBin;$env:PATH"
        $hostClangCl = Join-Path $llvmBin 'clang-cl.exe'
        $hostLldLink = Join-Path $llvmBin 'lld-link.exe'
    }

    if ($hostClangCl) {
        $hostCompiler = $hostClangCl
        $hostLinker = $hostLldLink
        $CommonCompilerFlags += ' -fuse-ld=lld'
        $CommonCMakeFlags += @(
            "LLVM_ENABLE_LLD=ON"
            "CMAKE_C_FLAGS=`"$CommonCompilerFlags`""
            "CMAKE_CXX_FLAGS=`"$CommonCompilerFlags`""
        )
    } else {
        # Pin the MSVC driver so bcrypt.lib is interpreted as a library.
        $hostCompiler = 'cl.exe'
        $hostLinker = 'link.exe'
    }
    Write-SubStep "Stage 0 host compiler: $hostCompiler (linker: $hostLinker)"

    $hostToolchainFlags = @(
        "CMAKE_C_COMPILER=$hostCompiler"
        "CMAKE_CXX_COMPILER=$hostCompiler"
        "CMAKE_LINKER=$hostLinker"
    )
    $CommonCMakeFlags += $hostToolchainFlags

    $env:VSCMD_START_DIR = $BuildDir
    $swigConfig = Find-Swig
    # CMake and Ninja subprocesses need the library when invoking SWIG.
    $env:SWIG_LIB = $swigConfig.LibraryDirectory

    # Directory names (always computed, used by multiple steps)
    $stage1Name = "build_${Arch}_stage1"
    $stage2Name = "build_${Arch}_stage2"
    $instrumentDir = Join-Path $BuildDir "instrument_${Arch}_stage2"
    $trainDir      = Join-Path $BuildDir "train_${Arch}_stage2"
    $profileFile   = Join-Path $BuildDir "profile_${Arch}_stage2.profdata"

    #-------------------------------------------------------------------
    # Step: libxml2
    #-------------------------------------------------------------------
    if (Test-ShouldRun 'libxml2') {
        # Every step that runs starts from a clean directory, so artifacts of
        # a previous run cannot leak into this one.
        Remove-StepDirectory (Join-Path $BuildDir $stage1Name 'libxmlbuild')
        Remove-StepDirectory (Join-Path $BuildDir $stage1Name 'zlibbuild')
        Remove-StepDirectory (Join-Path $BuildDir $stage1Name 'zstdbuild')
        New-Item -ItemType Directory -Path $stage1Name -Force | Out-Null
        Push-Location $stage1Name
        try {
            Build-ThirdPartyLibraries -Directory $ThirdPartyDir -ToolchainFlags $hostToolchainFlags
        } finally {
            Pop-Location
        }
    } else {
        Write-Step "Skipping libxml2 (-StartAt $($script:StartAtStep))"
        $libxmlInstallPath = Join-Path $BuildDir $stage1Name 'libxmlbuild' 'install'
        $zlibInstallPath = Join-Path $BuildDir $stage1Name 'zlibbuild' 'install'
        $zstdInstallPath = Join-Path $BuildDir $stage1Name 'zstdbuild' 'install'
        Assert-PathExists -Path $libxmlInstallPath -Description 'libxml2 install directory'
        Assert-PathExists -Path $zlibInstallPath -Description 'zlib install directory'
        Assert-PathExists -Path $zstdInstallPath -Description 'zstd install directory'
        $script:LibXmlInstallDir = ConvertTo-ForwardSlashPath $libxmlInstallPath
        $script:ZlibInstallDir = ConvertTo-ForwardSlashPath $zlibInstallPath
        $script:ZstdInstallDir = ConvertTo-ForwardSlashPath $zstdInstallPath
    }
    $libxmlDir = $script:LibXmlInstallDir
    $zlibDir = $script:ZlibInstallDir
    $zstdDir = $script:ZstdInstallDir

    # The stage 1 bootstrap tools are needed by PGO and stage 2.
    $stage1BinDir = ConvertTo-ForwardSlashPath (Join-Path $BuildDir $stage1Name 'bin')

    # Compute cmakeFlags (always needed, cheap -- derived from parameters)
    $cmakeFlags = $CommonCMakeFlags + @(
        "Python3_ROOT_DIR=$env:PYTHONHOME"
        "LIBXML2_INCLUDE_DIR=$libxmlDir/include/libxml2"
        "LIBXML2_LIBRARY=$libxmlDir/lib/libxml2s.lib"
        "LIBXML2_LIBRARIES=$libxmlDir/lib/libxml2s.lib;ws2_32.lib"
        "ZLIB_INCLUDE_DIR=$zlibDir/include"
        "ZLIB_LIBRARY=$zlibDir/lib/zs.lib"
        "ZLIB_LIBRARY_RELEASE=$zlibDir/lib/zs.lib"
        "zstd_INCLUDE_DIR=$zstdDir/include"
        "zstd_LIBRARY=$zstdDir/lib/zstd_static.lib"
    )

    $cmakeFlags += "CLANG_DEFAULT_LINKER=lld"
    if ($Arch -eq 'arm64') {
        $cmakeFlags += "COMPILER_RT_BUILD_SANITIZERS=OFF"
    }

    #-------------------------------------------------------------------
    # Step: stage 1 bootstrap compiler
    #-------------------------------------------------------------------
    if (Test-ShouldRun 'stage1') {
        # Preserve the dependency installs needed by stage 1.
        $stage1Dir = Join-Path $BuildDir $stage1Name
        if (Test-Path $stage1Dir) {
            Write-SubStep "Cleaning stage 1 bootstrap build artifacts (preserving dependency installs)..."
            Get-ChildItem -Path $stage1Dir -Exclude 'libxmlbuild', 'zlibbuild', 'zstdbuild' |
                ForEach-Object {
                    Remove-ItemWithoutProgress -LiteralPath $_.FullName -Recurse -Force
                }
        }

        # Stage 2 needs these tools even when the fast build skips the rest
        # of the stage 1 bootstrap build and its tests.
        $stage1InitialTargets = @('clang', 'lld', 'llvm-lib', 'llvm-rc', 'runtimes')
        if ($Arch -eq 'amd64') {
            $stage1InitialTargets += 'llvm-ml64'
        }
        $skipStage1FullBuild = $UseFastBuild
        $stage1Tests = @('check-llvm', 'check-clang', 'check-lld', 'check-runtimes')
        $stage1Extra = @("LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON",
                         "LLVM_TARGETS_TO_BUILD=Native")

        Build-Stage -Name $stage1Name -Description 'stage 1 bootstrap compiler' `
            -CMakeFlags $cmakeFlags `
            -LlvmSrc $LlvmSrc `
            -ExtraCMakeFlags $stage1Extra `
            -InitialBuildTargets $stage1InitialTargets `
            -SkipFullBuild:$skipStage1FullBuild `
            -TestTargets $stage1Tests `
            -Arch $Arch
    } else {
        Write-Step "Skipping stage 1 bootstrap compiler (-StartAt $($script:StartAtStep))"
        Assert-PathExists -Path "$stage1BinDir/clang-cl.exe" -Description 'stage 1 bootstrap clang-cl.exe'
        Assert-PathExists -Path "$stage1BinDir/lld-link.exe" -Description 'stage 1 bootstrap lld-link.exe'
        Assert-PathExists -Path "$stage1BinDir/llvm-lib.exe" -Description 'stage 1 bootstrap llvm-lib.exe'
        Assert-PathExists -Path "$stage1BinDir/llvm-rc.exe" -Description 'stage 1 bootstrap llvm-rc.exe'
        if ($Arch -eq 'amd64') {
            Assert-PathExists -Path "$stage1BinDir/llvm-ml64.exe" -Description 'stage 1 bootstrap llvm-ml64.exe'
        }
        # -FastBuild does not build llvm-profdata, which the PGO merge needs.
        if (-not $UseFastBuild -and (Test-ShouldRun 'pgo')) {
            Assert-PathExists -Path "$stage1BinDir/llvm-profdata.exe" -Description 'stage 1 bootstrap llvm-profdata.exe (not built by -FastBuild)'
        }
    }

    # Stage 2 is built with the stage 1 bootstrap toolchain.
    $stage2CompilerFlags = @(
        "CMAKE_C_COMPILER=$stage1BinDir/clang-cl.exe"
        "CMAKE_CXX_COMPILER=$stage1BinDir/clang-cl.exe"
        "CMAKE_LINKER=$stage1BinDir/lld-link.exe"
        "CMAKE_AR=$stage1BinDir/llvm-lib.exe"
        "CMAKE_RC_COMPILER=$stage1BinDir/llvm-rc.exe"
    )

    if ($Arch -eq 'amd64') {
        $stage2CompilerFlags += "CMAKE_ASM_MASM_COMPILER=$stage1BinDir/llvm-ml64.exe"
    } else {
        $stage2CompilerFlags += "CPACK_SYSTEM_NAME=woa64"
    }

    $stage2Flags = $cmakeFlags + $stage2CompilerFlags

    #-------------------------------------------------------------------
    # Step: pgo
    #-------------------------------------------------------------------
    $profileFlags = @()
    if (-not $UseFastBuild) {
        if (Test-ShouldRun 'pgo') {
            Remove-StepDirectory $instrumentDir
            Remove-StepDirectory $trainDir
            Remove-StepDirectory $profileFile
            New-PGOProfile -CMakeFlags $stage2Flags `
                -Stage1BinDir $stage1BinDir -LlvmSrc $LlvmSrc `
                -InstrumentDir $instrumentDir `
                -TrainDir $trainDir `
                -ProfilePath $profileFile
        } else {
            Write-Step "Skipping PGO (-StartAt $($script:StartAtStep))"
            Assert-PathExists -Path $profileFile -Description 'PGO profile data'
            $script:PGOProfilePath = ConvertTo-ForwardSlashPath $profileFile
        }
        $profilePath = $script:PGOProfilePath

        $pgoCFlags = "$CommonCompilerFlags -Wno-backend-plugin"
        $profileFlags = @(
            "LLVM_PROFDATA_FILE=$profilePath"
            "CMAKE_C_FLAGS=`"$pgoCFlags`""
            "CMAKE_CXX_FLAGS=`"$pgoCFlags`""
        )
    }

    #-------------------------------------------------------------------
    # Step: stage 2 self-hosted compiler
    #-------------------------------------------------------------------
    $stage2Projects = '"clang;clang-tools-extra;lld;lldb;flang;mlir"'

    $stage2Extra = @(
        "LLVM_ENABLE_PROJECTS=$stage2Projects"
        'LLVM_ENABLE_RUNTIMES="compiler-rt;openmp"'
        "PYTHON_HOME=$env:PYTHONHOME"
    ) + $CommonLLDBFlags + $Stage2CMakeFlags + @(
        "SWIG_EXECUTABLE=$($swigConfig.Executable)"
        "SWIG_DIR=$($swigConfig.LibraryDirectory)"
    ) + $profileFlags

    if (Test-ShouldRun 'stage2') {
        # Also discards the settings the tarball step left in the CMake cache.
        Remove-StepDirectory (Join-Path $BuildDir $stage2Name)

        $stage2Tests = @('check-llvm', 'check-clang', 'check-lld', 'check-runtimes',
                         'check-clang-tools', 'check-clangd')

        Build-Stage -Name $stage2Name -Description 'stage 2 self-hosted compiler' `
            -CMakeFlags $stage2Flags `
            -LlvmSrc $LlvmSrc `
            -ExtraCMakeFlags $stage2Extra `
            -TestTargets $stage2Tests `
            -Arch $Arch
    } else {
        Write-Step "Skipping stage 2 self-hosted compiler (-StartAt $($script:StartAtStep))"
        Assert-PathExists -Path (Join-Path $BuildDir $stage2Name) -Description 'stage 2 build directory'
    }

    $tripleArch = if ($Arch -eq 'amd64') { 'x86_64' } else { 'aarch64' }
    $filename = "clang+llvm-${PackageVersion}-${tripleArch}-pc-windows-msvc"
    $tarballInstallDirectory = Join-Path $BuildDir $filename

    #-------------------------------------------------------------------
    # Step: package (WiX MSI installer)
    #-------------------------------------------------------------------
    if (Test-ShouldRun 'package') {
        Write-Step "Creating WiX MSI installer package"
        Push-Location $stage2Name
        try {
            $tarballConfigMarker = Join-Path $PWD '.llvm_release_tarball_configured'
            if (Test-TarballConfigurationApplied -BuildDirectory $PWD.Path `
                    -TarballInstallDirectory $tarballInstallDirectory) {
                Write-SubStep 'The stage2 cache has tarball settings; restoring the MSI build configuration.'
                $packageFlags = $stage2Flags + $stage2Extra
                $cache = Write-CMakeCacheFile -Flags $packageFlags -FileName 'package_cache.cmake'
                $otherFlags = $cache.OtherFlags
                Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$LlvmSrc/llvm"
                Remove-Item -LiteralPath $tarballConfigMarker -Force -ErrorAction SilentlyContinue
            } else {
                Write-SubStep 'Stage2 is already configured for packaging; skipping CMake reconfigure.'
            }

            # CPack does not remove stale files from its staging tree, so
            # clean it in case a previous package attempt left staged files.
            Remove-StepDirectory (Join-Path $PWD '_CPack_Packages')

            Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs package
            # Downloaded release tags can predate these MSI settings. Check
            # each invariant only when the source being packaged defines it.
            $llvmCMakeLists = Join-Path $LlvmSrc 'llvm/CMakeLists.txt'
            if (Select-String -LiteralPath $llvmCMakeLists -Pattern 'set\(CPACK_WIX_UPGRADE_GUID' -Quiet) {
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
        Remove-StepDirectory (Join-Path $BuildDir $filename)
        Remove-StepDirectory (Join-Path $BuildDir "$filename.tar")
        Remove-StepDirectory (Join-Path $BuildDir "$filename.tar.xz")

        Push-Location $stage2Name
        try {
            $tarballFlags = $stage2Flags + $stage2Extra + @(
                "LLVM_INSTALL_TOOLCHAIN_ONLY=OFF"
                "CMAKE_INSTALL_PREFIX=$BuildDir/$filename"
                "LLVM_INCLUDE_TESTS=OFF"
            )
            $cache = Write-CMakeCacheFile -Flags $tarballFlags -FileName 'tarball_cache.cmake'
            $otherFlags = $cache.OtherFlags
            # Leave a durable marker so a later -StartAt package invocation
            # restores the stage2 settings before invoking CPack.
            Set-Content -LiteralPath '.llvm_release_tarball_configured' `
                -Value 'LLVM_INSTALL_TOOLCHAIN_ONLY=OFF' -Encoding ASCII
            Invoke-NativeCommand cmake -GNinja -C $cache.CacheFile @otherFlags "$LlvmSrc/llvm"
            Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs install

            # Verify llvm-config works
            Invoke-NativeCommand "$BuildDir/$filename/bin/llvm-config.exe" --bindir
        } finally {
            Pop-Location
        }

        # Create the portable archive without changing executable layout.
        Invoke-NativeCommand 7z a -ttar "$filename.tar" $filename
        Invoke-NativeCommand 7z a -txz "$filename.tar.xz" "$filename.tar"
        Remove-Item -LiteralPath "$filename.tar"
    } else {
        Write-Step "Skipping tarball (-StartAt $($script:StartAtStep))"
    }
}

