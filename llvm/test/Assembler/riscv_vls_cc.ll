; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: define riscv_vls_cc(32) void @no_args_32() {
define riscv_vls_cc(32) void @no_args_32() {
  ret void
}

; CHECK: define riscv_vls_cc(64) void @no_args_64() {
define riscv_vls_cc(64) void @no_args_64() {
  ret void
}

; CHECK: define riscv_vls_cc(128) void @no_args_128() {
define riscv_vls_cc(128) void @no_args_128() {
  ret void
}

; CHECK: define riscv_vls_cc(256) void @no_args_256() {
define riscv_vls_cc(256) void @no_args_256() {
  ret void
}

; CHECK: define riscv_vls_cc(512) void @no_args_512() {
define riscv_vls_cc(512) void @no_args_512() {
  ret void
}

; CHECK: define riscv_vls_cc(1024) void @no_args_1024() {
define riscv_vls_cc(1024) void @no_args_1024() {
  ret void
}

; CHECK: define riscv_vls_cc(2048) void @no_args_2048() {
define riscv_vls_cc(2048) void @no_args_2048() {
  ret void
}

; CHECK: define riscv_vls_cc(4096) void @no_args_4096() {
define riscv_vls_cc(4096) void @no_args_4096() {
  ret void
}

; CHECK: define riscv_vls_cc(8192) void @no_args_8192() {
define riscv_vls_cc(8192) void @no_args_8192() {
  ret void
}

; CHECK: define riscv_vls_cc(16384) void @no_args_16384() {
define riscv_vls_cc(16384) void @no_args_16384() {
  ret void
}

; CHECK: define riscv_vls_cc(32768) void @no_args_32768() {
define riscv_vls_cc(32768) void @no_args_32768() {
  ret void
}

; CHECK: define riscv_vls_cc(65536) void @no_args_65536() {
define riscv_vls_cc(65536) void @no_args_65536() {
  ret void
}

; CHECK: define riscv_vls_cc(32) void @byval_arg_32(ptr byval(i32) %0) {
define riscv_vls_cc(32) void @byval_arg_32(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(64) void @byval_arg_64(ptr byval(i32) %0) {
define riscv_vls_cc(64) void @byval_arg_64(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(128) void @byval_arg_128(ptr byval(i32) %0) {
define riscv_vls_cc(128) void @byval_arg_128(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(256) void @byval_arg_256(ptr byval(i32) %0) {
define riscv_vls_cc(256) void @byval_arg_256(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(512) void @byval_arg_512(ptr byval(i32) %0) {
define riscv_vls_cc(512) void @byval_arg_512(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(1024) void @byval_arg_1024(ptr byval(i32) %0) {
define riscv_vls_cc(1024) void @byval_arg_1024(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(2048) void @byval_arg_2048(ptr byval(i32) %0) {
define riscv_vls_cc(2048) void @byval_arg_2048(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(4096) void @byval_arg_4096(ptr byval(i32) %0) {
define riscv_vls_cc(4096) void @byval_arg_4096(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(8192) void @byval_arg_8192(ptr byval(i32) %0) {
define riscv_vls_cc(8192) void @byval_arg_8192(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(16384) void @byval_arg_16384(ptr byval(i32) %0) {
define riscv_vls_cc(16384) void @byval_arg_16384(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(32768) void @byval_arg_32768(ptr byval(i32) %0) {
define riscv_vls_cc(32768) void @byval_arg_32768(ptr byval(i32)) {
  ret void
}

; CHECK: define riscv_vls_cc(65536) void @byval_arg_65536(ptr byval(i32) %0) {
define riscv_vls_cc(65536) void @byval_arg_65536(ptr byval(i32)) {
  ret void
}
