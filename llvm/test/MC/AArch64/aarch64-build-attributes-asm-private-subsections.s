// RUN: llvm-mc -triple=aarch64 %s -o - | FileCheck %s --check-prefix=CHECK

.aeabi_subsection private_subsection_1, optional, uleb128
.aeabi_attribute 12, 257
// CHECK: .aeabi_subsection	private_subsection_1, optional, uleb128
// CHECK: .aeabi_attribute 12, 257

.aeabi_subsection private_subsection_2, required, uleb128
.aeabi_attribute 76, 257
// CHECK: .aeabi_subsection	private_subsection_2, required, uleb128
// CHECK: .aeabi_attribute 76, 257

.aeabi_subsection private_subsection_3, optional, ntbs
.aeabi_attribute 34, hello_llvm
// CHECK: .aeabi_subsection	private_subsection_3, optional, ntbs
// CHECK: .aeabi_attribute 34, hello_llvm

.aeabi_subsection private_subsection_4, required, ntbs
.aeabi_attribute 777, hello_llvm
// CHECK: .aeabi_subsection	private_subsection_4, required, ntbs
// CHECK: .aeabi_attribute 777, hello_llvm

.aeabi_subsection private_subsection_1, optional, uleb128
.aeabi_attribute 876, 257
// CHECK: .aeabi_subsection	private_subsection_1, optional, uleb128
// CHECK: .aeabi_attribute 876, 257

.aeabi_subsection private_subsection_2, required, uleb128
.aeabi_attribute 876, 257
// CHECK: .aeabi_subsection	private_subsection_2, required, uleb128
// CHECK: .aeabi_attribute 876, 257

.aeabi_subsection private_subsection_3, optional, ntbs
.aeabi_attribute 876, hello_llvm
// CHECK: .aeabi_subsection private_subsection_3, optional, ntbs
// CHECK: .aeabi_attribute 876, hello_llvm

.aeabi_subsection private_subsection_4, required, ntbs
.aeabi_attribute 876, hello_llvm
// CHECK: .aeabi_subsection	private_subsection_4, required, ntbs
// CHECK: .aeabi_attribute 876, hello_llvm