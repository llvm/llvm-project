// clang-format off
// REQUIRES: lld, x86

// Test that relocatable PE/COFF binaries (like UEFI drivers) with PDB symbols
// can be loaded. These binaries return LLDB_INVALID_ADDRESS for
// GetFileAddress() and should use 0x0 as the base to allow RVA-based symbol
// loading.

// RUN: %clang_cl --target=x86_64-unknown-uefi -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:EfiMain -subsystem:efi_application %t.obj -out:%t.efi -pdb:%t.pdb
// RUN: lldb-test symbols %t.efi | FileCheck %s

// CHECK: Function{{{.*}}}, mangled = ?EfiMain@@YAHXZ

int EfiMain() {
  return 0;
}
