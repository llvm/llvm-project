// RUN: llvm-mc %s -triple x86_64-apple-tvos -filetype=obj | llvm-readobj -macho-version-min | FileCheck %s

.build_version tvos,1,2,3
// CHECK: MinVersion {
// CHECK:   Cmd: LC_BUILD_VERSION
// CHECK:   Size: 24
// CHECK:   Platform: tvos
// CHECK:   Version: 1.2.3
// CHECK:   SDK: n/a
// CHECK: }
