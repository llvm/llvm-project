; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx600 < %s | FileCheck --check-prefixes=GFX600 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tahiti < %s | FileCheck --check-prefixes=GFX600 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx601 < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=pitcairn < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=verde < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx602 < %s | FileCheck --check-prefixes=GFX602 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hainan < %s | FileCheck --check-prefixes=GFX602 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=oland < %s | FileCheck --check-prefixes=GFX602 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 < %s | FileCheck --check-prefixes=GFX700 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri < %s | FileCheck --check-prefixes=GFX700 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx701 < %s | FileCheck --check-prefixes=GFX701 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii < %s | FileCheck --check-prefixes=GFX701 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx702 < %s | FileCheck --check-prefixes=GFX702 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx703 < %s | FileCheck --check-prefixes=GFX703 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kabini < %s | FileCheck --check-prefixes=GFX703 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=mullins < %s | FileCheck --check-prefixes=GFX703 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx704 < %s | FileCheck --check-prefixes=GFX704 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=bonaire < %s | FileCheck --check-prefixes=GFX704 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx705 < %s | FileCheck --check-prefixes=GFX705 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx801 < %s | FileCheck --check-prefixes=GFX801 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx801 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX801-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx801 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX801-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=carrizo < %s | FileCheck --check-prefixes=GFX801 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=carrizo -mattr=-xnack < %s | FileCheck --check-prefixes=GFX801-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=carrizo -mattr=+xnack < %s | FileCheck --check-prefixes=GFX801-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 < %s | FileCheck --check-prefixes=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=iceland < %s | FileCheck --check-prefixes=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tonga < %s | FileCheck --check-prefixes=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=polaris10 < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=polaris11 < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx805 < %s | FileCheck --check-prefixes=GFX805 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tongapro < %s | FileCheck --check-prefixes=GFX805 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx810 < %s | FileCheck --check-prefixes=GFX810 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx810 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX810-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx810 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX810-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=stoney < %s | FileCheck --check-prefixes=GFX810 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=stoney -mattr=-xnack < %s | FileCheck --check-prefixes=GFX810-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=stoney -mattr=+xnack < %s | FileCheck --check-prefixes=GFX810-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck --check-prefixes=GFX900 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX900-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX900-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx902 < %s | FileCheck --check-prefixes=GFX902 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx902 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX902-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx902 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX902-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx904 < %s | FileCheck --check-prefixes=GFX904 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx904 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX904-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx904 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX904-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 < %s | FileCheck --check-prefixes=GFX906 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=-sramecc < %s | FileCheck --check-prefixes=GFX906-NOSRAMECC %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+sramecc < %s | FileCheck --check-prefixes=GFX906-SRAMECC %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX906-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX906-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=-sramecc,-xnack < %s | FileCheck --check-prefixes=GFX906-NOSRAMECC-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+sramecc,-xnack < %s | FileCheck --check-prefixes=GFX906-SRAMECC-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=-sramecc,+xnack < %s | FileCheck --check-prefixes=GFX906-NOSRAMECC-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+sramecc,+xnack < %s | FileCheck --check-prefixes=GFX906-SRAMECC-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 < %s | FileCheck --check-prefixes=GFX908 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-sramecc < %s | FileCheck --check-prefixes=GFX908-NOSRAMECC %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+sramecc < %s | FileCheck --check-prefixes=GFX908-SRAMECC %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX908-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX908-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-sramecc,-xnack < %s | FileCheck --check-prefixes=GFX908-NOSRAMECC-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+sramecc,-xnack < %s | FileCheck --check-prefixes=GFX908-SRAMECC-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-sramecc,+xnack < %s | FileCheck --check-prefixes=GFX908-NOSRAMECC-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+sramecc,+xnack < %s | FileCheck --check-prefixes=GFX908-SRAMECC-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx909 < %s | FileCheck --check-prefixes=GFX909 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx909 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX909-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx909 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX909-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90c < %s | FileCheck --check-prefixes=GFX90C %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90c -mattr=-xnack < %s | FileCheck --check-prefixes=GFX90C-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90c -mattr=+xnack < %s | FileCheck --check-prefixes=GFX90C-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s | FileCheck --check-prefixes=GFX942 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX942-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX942-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 < %s | FileCheck --check-prefixes=GFX950 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX950-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX950-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 < %s | FileCheck --check-prefixes=GFX1010 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX1010-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX1010-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1011 < %s | FileCheck --check-prefixes=GFX1011 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1011 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX1011-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1011 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX1011-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1012 < %s | FileCheck --check-prefixes=GFX1012 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1012 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX1012-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1012 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX1012-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1013 < %s | FileCheck --check-prefixes=GFX1013 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1013 -mattr=-xnack < %s | FileCheck --check-prefixes=GFX1013-NOXNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1013 -mattr=+xnack < %s | FileCheck --check-prefixes=GFX1013-XNACK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 < %s | FileCheck --check-prefixes=GFX1030 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1031 < %s | FileCheck --check-prefixes=GFX1031 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1032 < %s | FileCheck --check-prefixes=GFX1032 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1033 < %s | FileCheck --check-prefixes=GFX1033 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1034 < %s | FileCheck --check-prefixes=GFX1034 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1035 < %s | FileCheck --check-prefixes=GFX1035 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1036 < %s | FileCheck --check-prefixes=GFX1036 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s | FileCheck --check-prefixes=GFX1100 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1101 < %s | FileCheck --check-prefixes=GFX1101 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1102 < %s | FileCheck --check-prefixes=GFX1102 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1103 < %s | FileCheck --check-prefixes=GFX1103 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 < %s | FileCheck --check-prefixes=GFX1150 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1151 < %s | FileCheck --check-prefixes=GFX1151 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1152 < %s | FileCheck --check-prefixes=GFX1152 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1153 < %s | FileCheck --check-prefixes=GFX1153 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 < %s | FileCheck --check-prefixes=GFX1200 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 < %s | FileCheck --check-prefixes=GFX1201 %s

; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx9-generic -mattr=-xnack < %s | FileCheck --check-prefixes=GFX9_GENERIC_NOXNACK %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx9-generic -mattr=+xnack < %s | FileCheck --check-prefixes=GFX9_GENERIC_XNACK %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx9-4-generic -mattr=-xnack < %s | FileCheck --check-prefixes=GFX9_4_GENERIC_NOXNACK %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx9-4-generic -mattr=+xnack < %s | FileCheck --check-prefixes=GFX9_4_GENERIC_XNACK %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-1-generic -mattr=-xnack < %s | FileCheck --check-prefixes=GFX10_1_GENERIC_NOXNACK %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-1-generic -mattr=+xnack < %s | FileCheck --check-prefixes=GFX10_1_GENERIC_XNACK %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-3-generic < %s | FileCheck --check-prefixes=GFX10_3_GENERIC %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx11-generic < %s | FileCheck --check-prefixes=GFX11_GENERIC %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx12-generic < %s | FileCheck --check-prefixes=GFX12_GENERIC %s

; GFX600: .amdgcn_target "amdgcn-amd-amdhsa--gfx600"
; GFX601: .amdgcn_target "amdgcn-amd-amdhsa--gfx601"
; GFX602: .amdgcn_target "amdgcn-amd-amdhsa--gfx602"
; GFX700: .amdgcn_target "amdgcn-amd-amdhsa--gfx700"
; GFX701: .amdgcn_target "amdgcn-amd-amdhsa--gfx701"
; GFX702: .amdgcn_target "amdgcn-amd-amdhsa--gfx702"
; GFX703: .amdgcn_target "amdgcn-amd-amdhsa--gfx703"
; GFX704: .amdgcn_target "amdgcn-amd-amdhsa--gfx704"
; GFX705: .amdgcn_target "amdgcn-amd-amdhsa--gfx705"
; GFX801: .amdgcn_target "amdgcn-amd-amdhsa--gfx801"
; GFX801-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx801:xnack-"
; GFX801-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx801:xnack+"
; GFX802: .amdgcn_target "amdgcn-amd-amdhsa--gfx802"
; GFX803: .amdgcn_target "amdgcn-amd-amdhsa--gfx803"
; GFX805: .amdgcn_target "amdgcn-amd-amdhsa--gfx805"
; GFX810: .amdgcn_target "amdgcn-amd-amdhsa--gfx810"
; GFX810-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx810:xnack-"
; GFX810-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx810:xnack+"
; GFX900: .amdgcn_target "amdgcn-amd-amdhsa--gfx900"
; GFX900-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx900:xnack-"
; GFX900-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx900:xnack+"
; GFX902: .amdgcn_target "amdgcn-amd-amdhsa--gfx902"
; GFX902-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx902:xnack-"
; GFX902-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx902:xnack+"
; GFX904: .amdgcn_target "amdgcn-amd-amdhsa--gfx904"
; GFX904-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx904:xnack-"
; GFX904-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx904:xnack+"
; GFX906: .amdgcn_target "amdgcn-amd-amdhsa--gfx906"
; GFX906-NOSRAMECC: .amdgcn_target "amdgcn-amd-amdhsa--gfx906:sramecc-"
; GFX906-SRAMECC: .amdgcn_target "amdgcn-amd-amdhsa--gfx906:sramecc+"
; GFX906-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx906:xnack-"
; GFX906-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx906:xnack+"
; GFX906-NOSRAMECC-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx906:sramecc-:xnack-"
; GFX906-SRAMECC-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx906:sramecc+:xnack-"
; GFX906-NOSRAMECC-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx906:sramecc-:xnack+"
; GFX906-SRAMECC-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx906:sramecc+:xnack+"
; GFX908: .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
; GFX908-NOSRAMECC: .amdgcn_target "amdgcn-amd-amdhsa--gfx908:sramecc-"
; GFX908-SRAMECC: .amdgcn_target "amdgcn-amd-amdhsa--gfx908:sramecc+"
; GFX908-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx908:xnack-"
; GFX908-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx908:xnack+"
; GFX908-NOSRAMECC-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx908:sramecc-:xnack-"
; GFX908-SRAMECC-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx908:sramecc+:xnack-"
; GFX908-NOSRAMECC-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx908:sramecc-:xnack+"
; GFX908-SRAMECC-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx908:sramecc+:xnack+"
; GFX909: .amdgcn_target "amdgcn-amd-amdhsa--gfx909"
; GFX909-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx909:xnack-"
; GFX909-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx909:xnack+"
; GFX90C: .amdgcn_target "amdgcn-amd-amdhsa--gfx90c"
; GFX90C-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx90c:xnack-"
; GFX90C-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx90c:xnack+"
; GFX942: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
; GFX942-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942:xnack-"
; GFX942-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942:xnack+"
; GFX950: .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
; GFX950-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx950:xnack-"
; GFX950-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx950:xnack+"
; GFX1010: .amdgcn_target "amdgcn-amd-amdhsa--gfx1010"
; GFX1010-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1010:xnack-"
; GFX1010-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1010:xnack+"
; GFX1011: .amdgcn_target "amdgcn-amd-amdhsa--gfx1011"
; GFX1011-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1011:xnack-"
; GFX1011-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1011:xnack+"
; GFX1012: .amdgcn_target "amdgcn-amd-amdhsa--gfx1012"
; GFX1012-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1012:xnack-"
; GFX1012-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1012:xnack+"
; GFX1013: .amdgcn_target "amdgcn-amd-amdhsa--gfx1013"
; GFX1013-NOXNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1013:xnack-"
; GFX1013-XNACK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1013:xnack+"
; GFX1030: .amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
; GFX1031: .amdgcn_target "amdgcn-amd-amdhsa--gfx1031"
; GFX1032: .amdgcn_target "amdgcn-amd-amdhsa--gfx1032"
; GFX1033: .amdgcn_target "amdgcn-amd-amdhsa--gfx1033"
; GFX1034: .amdgcn_target "amdgcn-amd-amdhsa--gfx1034"
; GFX1035: .amdgcn_target "amdgcn-amd-amdhsa--gfx1035"
; GFX1036: .amdgcn_target "amdgcn-amd-amdhsa--gfx1036"
; GFX1100: .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
; GFX1101: .amdgcn_target "amdgcn-amd-amdhsa--gfx1101"
; GFX1102: .amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
; GFX1103: .amdgcn_target "amdgcn-amd-amdhsa--gfx1103"
; GFX1150: .amdgcn_target "amdgcn-amd-amdhsa--gfx1150"
; GFX1151: .amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
; GFX1152: .amdgcn_target "amdgcn-amd-amdhsa--gfx1152"
; GFX1153: .amdgcn_target "amdgcn-amd-amdhsa--gfx1153"
; GFX1200: .amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
; GFX1201: .amdgcn_target "amdgcn-amd-amdhsa--gfx1201"

; GFX9_GENERIC_NOXNACK:     .amdgcn_target "amdgcn-amd-amdhsa--gfx9-generic:xnack-"
; GFX9_GENERIC_XNACK:       .amdgcn_target "amdgcn-amd-amdhsa--gfx9-generic:xnack+"
; GFX9_4_GENERIC_NOXNACK:   .amdgcn_target "amdgcn-amd-amdhsa--gfx9-4-generic:xnack-"
; GFX9_4_GENERIC_XNACK:     .amdgcn_target "amdgcn-amd-amdhsa--gfx9-4-generic:xnack+"
; GFX10_1_GENERIC_NOXNACK:  .amdgcn_target "amdgcn-amd-amdhsa--gfx10-1-generic:xnack-"
; GFX10_1_GENERIC_XNACK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx10-1-generic:xnack+"
; GFX10_3_GENERIC:          .amdgcn_target "amdgcn-amd-amdhsa--gfx10-3-generic"
; GFX11_GENERIC:            .amdgcn_target "amdgcn-amd-amdhsa--gfx11-generic"
; GFX12_GENERIC:            .amdgcn_target "amdgcn-amd-amdhsa--gfx12-generic"

define amdgpu_kernel void @directive_amdgcn_target() {
  ret void
}
