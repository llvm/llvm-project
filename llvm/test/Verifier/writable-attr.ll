; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: Attribute 'writable' applied to incompatible type!
; CHECK-NEXT: ptr @not_pointer
define void @not_pointer_writable(i32 writable %arg) {
  ret void
}

; CHECK: Attributes writable and readnone are incompatible!
; CHECK-NEXT: ptr @writable_readnone
define void @writable_readnone(ptr writable readnone %arg) {
  ret void
}

; CHECK: Attributes writable and readonly are incompatible!
; CHECK-NEXT: ptr @writable_readonly
define void @writable_readonly(ptr writable readonly %arg) {
  ret void
}

; CHECK: Attribute writable and memory without argmem: write are incompatible!
; CHECK-NEXT: ptr @writable_memory_argmem_read
define void @writable_memory_argmem_read(ptr writable %arg) memory(write, argmem: read) {
  ret void
}

; CHECK-NOT: incompatible
define void @writable_memory_argmem_write(ptr writable %arg) memory(read, argmem: write) {
  ret void
}
