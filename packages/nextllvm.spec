%define _binary_payload w9.gzdio
%define _build_id_links none
AutoReqProv: no
Name:       {{PKG_NAME}}
Version:    {{MAJOR_VERSION}}
Release:    {{RELEASE}}
Vendor:     NextSilicon
Summary:    NextSilicon's LLVM
URL:        www.nextsilicon.com
Requires:   {{DEPENDENCIES}}
BuildArch:  x86_64
License:    Commercial
Packager:   NextSilicon Support <support@nextsilicon.com>

%description
NextSilicon's LLVM
 Branch: {{BRANCH}}
 GIT_SHA: {{GIT_SHA}}

%define _unpackaged_files_terminate_build 0

%files
%defattr(-, root, root, -)
/opt/nextsilicon/bin/*
/opt/nextsilicon/include/*
/opt/nextsilicon/lib/*
/opt/nextsilicon/llvm/*
/opt/nextsilicon/riscv/*
/opt/nextsilicon/sysroot/*
