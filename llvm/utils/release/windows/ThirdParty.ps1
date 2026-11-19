# Third-party dependency build helpers.

# CMakeFlags holds each library's own settings. The settings shared by every
# library are in Build-ThirdPartyLibrary.
$script:ThirdPartyLibraries = @(
    @{
        Name = 'libxml2'
        SourceDirectory = 'libxml2-v2.15.3'
        Archive = 'libxml2-v2.15.3.tar.gz'
        DownloadArguments = @('--remote-name', 'https://gitlab.gnome.org/GNOME/libxml2/-/archive/v2.15.3/libxml2-v2.15.3.tar.gz')
        Sha256 = '0DA50C1415F4EC0364569D2119B1436BA837B31DF44AF28569D234272C23CF1F'
        ExtractArguments = @('zxf', 'libxml2-v2.15.3.tar.gz', '--exclude', 'test/*')
        BuildDirectory = 'libxmlbuild'
        CMakeFlags = @(
            'BUILD_SHARED_LIBS=OFF'
            'LIBXML2_WITH_C14N=OFF'
            'LIBXML2_WITH_CATALOG=OFF'
            'LIBXML2_WITH_DEBUG=OFF'
            'LIBXML2_WITH_DOCB=OFF'
            'LIBXML2_WITH_FTP=OFF'
            'LIBXML2_WITH_HTML=OFF'
            'LIBXML2_WITH_HTTP=OFF'
            'LIBXML2_WITH_ICONV=OFF'
            'LIBXML2_WITH_ICU=OFF'
            'LIBXML2_WITH_ISO8859X=OFF'
            'LIBXML2_WITH_LEGACY=OFF'
            'LIBXML2_WITH_LZMA=OFF'
            'LIBXML2_WITH_MEM_DEBUG=OFF'
            'LIBXML2_WITH_MODULES=OFF'
            'LIBXML2_WITH_OUTPUT=ON'
            'LIBXML2_WITH_PATTERN=OFF'
            'LIBXML2_WITH_PROGRAMS=OFF'
            'LIBXML2_WITH_PUSH=OFF'
            'LIBXML2_WITH_PYTHON=OFF'
            'LIBXML2_WITH_READER=OFF'
            'LIBXML2_WITH_REGEXPS=OFF'
            'LIBXML2_WITH_RUN_DEBUG=OFF'
            'LIBXML2_WITH_SAX1=ON'
            'LIBXML2_WITH_SCHEMAS=OFF'
            'LIBXML2_WITH_SCHEMATRON=OFF'
            'LIBXML2_WITH_TESTS=OFF'
            'LIBXML2_WITH_THREADS=ON'
            'LIBXML2_WITH_THREAD_ALLOC=OFF'
            'LIBXML2_WITH_TREE=ON'
            'LIBXML2_WITH_VALID=OFF'
            'LIBXML2_WITH_WRITER=OFF'
            'LIBXML2_WITH_XINCLUDE=OFF'
            'LIBXML2_WITH_XPATH=OFF'
            'LIBXML2_WITH_XPTR=OFF'
            'LIBXML2_WITH_ZLIB=OFF'
        )
    }
    @{
        Name = 'zlib'
        SourceDirectory = 'zlib-1.3.2'
        Archive = 'zlib-1.3.2.tar.gz'
        DownloadArguments = @('-LO', 'https://github.com/madler/zlib/releases/download/v1.3.2/zlib-1.3.2.tar.gz')
        Sha256 = 'BB329A0A2CD0274D05519D61C667C062E06990D72E125EE2DFA8DE64F0119D16'
        ExtractArguments = @('zxf', 'zlib-1.3.2.tar.gz')
        BuildDirectory = 'zlibbuild'
        CMakeFlags = @(
            'ZLIB_BUILD_TESTING=OFF'
            'ZLIB_BUILD_SHARED=OFF'
            'ZLIB_BUILD_STATIC=ON'
            'ZLIB_INSTALL=ON'
        )
    }
    @{
        Name = 'zstd'
        SourceDirectory = 'zstd-1.5.7'
        Archive = 'zstd-1.5.7.tar.gz'
        DownloadArguments = @('-LO', 'https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz')
        Sha256 = 'EB33E51F49A15E023950CD7825CA74A4A2B43DB8354825AC24FC1B7EE09E6FA3'
        ExtractArguments = @('zxf', 'zstd-1.5.7.tar.gz', '--exclude', 'tests/*')
        BuildDirectory = 'zstdbuild'
        # zstd's CMakeLists.txt is not at the top of its source tree.
        CMakeListsDirectory = 'build/cmake'
        CMakeFlags = @(
            'CMAKE_CXX_FLAGS=-w'
            'ZSTD_BUILD_PROGRAMS=ON'
            'ZSTD_BUILD_TESTS=OFF'
            'ZSTD_BUILD_STATIC=ON'
            'ZSTD_BUILD_SHARED=OFF'
            # zstd pins its CMake policies to version 3.13, which predates
            # CMP0091, so CMAKE_MSVC_RUNTIME_LIBRARY needs it enabled here.
            'CMAKE_POLICY_DEFAULT_CMP0091=NEW'
        )
    }
)

# Install directory of each library, by name. Build-ThirdPartyLibrary sets it.
$script:ThirdPartyInstallDirs = @{}

function Install-ThirdPartySources {
    param(
        [Parameter(Mandatory)][string]$Directory,
        [switch]$SkipExisting
    )

    Push-Location $Directory
    try {
        foreach ($Library in $script:ThirdPartyLibraries) {
            if ($SkipExisting -and (Test-Path -LiteralPath $Library.SourceDirectory)) {
                Write-Step "Skipping $($Library.Name) download (already exists)"
                continue
            }

            Write-Step "Downloading $($Library.Name)"
            $DownloadArguments = $Library.DownloadArguments
            Invoke-NativeCommand curl.exe @DownloadArguments
            Test-FileChecksum -Path $Library.Archive -Algorithm SHA256 `
                -ExpectedHash $Library.Sha256
            $ExtractArguments = $Library.ExtractArguments
            Invoke-NativeCommand tar @ExtractArguments
        }
    } finally {
        Pop-Location
    }
}

function Build-ThirdPartyLibrary {
    <#
    .SYNOPSIS
        Builds and installs one library from $script:ThirdPartyLibraries, and
        sets its entry in $script:ThirdPartyInstallDirs to the install path.
    .NOTES
        Uses a script-scoped variable instead of return to avoid PowerShell
        capturing native command stdout as part of the function's return value,
        which would break Ninja's \r progress display and corrupt the path.
    #>
    param(
        [Parameter(Mandatory)]
        [hashtable]$Library,
        [Parameter(Mandatory)]
        [string]$SourceDir,
        [Parameter(Mandatory)]
        [string]$BuildDirectory,
        [Parameter(Mandatory)]
        [string[]]$ToolchainFlags
    )
    $Name = $Library.Name
    # Skipped steps leave the previous install in place. Callers can invoke
    # this unconditionally.
    if (-not (Test-ShouldRun 'dependencies')) {
        $InstallDirectory = Join-Path $BuildDirectory 'install'
        Write-SubStep "Skipping $Name (-StartAt $($script:StartAtStep))"
        Assert-PathExists -Path $InstallDirectory -Description "$Name install directory"
        $script:ThirdPartyInstallDirs[$Name] = ConvertTo-UnixPath $InstallDirectory
        return
    }
    Write-SubStep "Building $Name..."
    Remove-StepDirectory $BuildDirectory
    New-Item -ItemType Directory -Path $BuildDirectory -Force | Out-Null
    Push-Location $BuildDirectory
    try {
        $InstallDirectory = Join-Path $PWD 'install'
        $Flags = $ToolchainFlags + @(
            "CMAKE_BUILD_TYPE=Release"
            # Suppress warnings in the downloaded dependencies with -w; clang,
            # clang-cl and cl all accept it.
            "CMAKE_C_FLAGS=-w"
            "CMAKE_INSTALL_PREFIX=$InstallDirectory"
            "CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
        ) + $Library.CMakeFlags
        $CacheFile = Write-CMakeCacheFile -Flags $Flags -FileName "${Name}_cache.cmake"
        # Strict mode rejects reading a missing key as a property.
        $CMakeSourceDir = if ($Library.ContainsKey('CMakeListsDirectory')) {
            "$SourceDir/$($Library.CMakeListsDirectory)"
        } else {
            $SourceDir
        }
        Invoke-NativeCommand cmake -G Ninja -C $CacheFile $CMakeSourceDir
        Invoke-NativeCommand $script:NinjaCommand @script:NinjaExtraArgs install
        $script:ThirdPartyInstallDirs[$Name] = ConvertTo-UnixPath $InstallDirectory
    } finally {
        Pop-Location
    }
}
