; RUN: llc %s -march=mipsel -mattr=micromips -filetype=asm \
; RUN: -relocation-model=pic -O3 -o - | FileCheck %s

; Function Attrs: noinline nounwind
define void @bar(ptr %p) #0 {
entry:
  %p.addr = alloca ptr, align 4
  store ptr %p, ptr %p.addr, align 4
  %0 = load ptr, ptr %p.addr, align 4
  %1 = load i32, ptr %0, align 4
  %add = add nsw i32 7, %1
  %2 = load ptr, ptr %p.addr, align 4
  store i32 %add, ptr %2, align 4
  %3 = load ptr, ptr %p.addr, align 4
  %add.ptr = getelementptr inbounds i32, ptr %3, i32 1
  %4 = load i32, ptr %add.ptr, align 4
  %add1 = add nsw i32 7, %4
  %5 = load ptr, ptr %p.addr, align 4
  %add.ptr2 = getelementptr inbounds i32, ptr %5, i32 1
  store i32 %add1, ptr %add.ptr2, align 4
  ret void
}

; CHECK: lw16 ${{[0-9]+}}, 0($4)
; CHECK: sw16 ${{[0-9]+}}, 0($4)
; CHECK: lw16 ${{[0-9]+}}, 4(${{[0-9]+}})
; CHECK: sw16 ${{[0-9]+}}, 4(${{[0-9]+}})
