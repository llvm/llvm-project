# escape=`

FROM mcr.microsoft.com/windows/servercore:ltsc2019

LABEL maintainer "LLVM Developers"

# Set cmd as default shell
SHELL ["cmd", "/S", "/C"]

# Setup vs_buildtools.exe
COPY Install.cmd C:\TEMP\
ADD https://aka.ms/vscollect.exe C:\TEMP\collect.exe
ADD https://aka.ms/vs/16/release/channel C:\TEMP\VisualStudio.chman
ADD https://aka.ms/vs/16/release/vs_buildtools.exe C:\TEMP\vs_buildtools.exe

# Install VS2019 x86 x64 C++
RUN C:\TEMP\Install.cmd C:\TEMP\vs_buildtools.exe --quiet --wait --norestart --nocache `
    --installPath C:\BuildTools `
    --channelUri C:\TEMP\VisualStudio.chman `
    --installChannelUri C:\TEMP\VisualStudio.chman `
    --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64

# Install Windows 10 SDK 19041 (apparently this is needed to let CMake detect MSVC??)
RUN C:\TEMP\Install.cmd C:\TEMP\vs_buildtools.exe --quiet --wait --norestart --nocache `
    --installPath C:\BuildTools `
    --channelUri C:\TEMP\VisualStudio.chman `
    --installChannelUri C:\TEMP\VisualStudio.chman `
    --add Microsoft.VisualStudio.Component.Windows10SDK.19041

# Install VS CMake
RUN C:\TEMP\Install.cmd C:\TEMP\vs_buildtools.exe --quiet --wait --norestart --nocache `
    --installPath C:\BuildTools `
    --channelUri C:\TEMP\VisualStudio.chman `
    --installChannelUri C:\TEMP\VisualStudio.chman `
    --add Microsoft.VisualStudio.Component.VC.CMake.Project

# Add CMake to PATH
RUN powershell setx /M PATH $($Env:PATH + ';C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin')


# Install Chocolatey
RUN powershell -Command `
        iex ((new-object net.webclient).DownloadString('https://chocolatey.org/install.ps1')); `
        choco feature disable --name showDownloadProgress

# To determine version
RUN choco install -y git

RUN choco install -y python --version=3.9.0