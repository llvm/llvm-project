; RUN: llc -march=mipsel -mcpu=mips32r6  -disable-mips-delay-filler < %s | FileCheck %s
; RUN: llc -march=mips -mcpu=mips32r6 -disable-mips-delay-filler < %s \
; RUN:   -filetype=obj -o - | llvm-objdump -d - | FileCheck %s -check-prefix=ENCODING

; bnezc and beqzc have restriction that $rt != 0

define i32 @f() {
; CHECK-LABEL: f
; CHECK-NOT:   bnezc $0

  %cmp = icmp eq i32 1, 1
  br i1 %cmp, label %if.then, label %if.end

  if.then:
    ret i32 1

  if.end:
    ret i32 0
}

define i32 @f1() {
; CHECK-LABEL: f1
; CHECK-NOT:   beqzc $0

  %cmp = icmp eq i32 0, 0
  br i1 %cmp, label %if.then, label %if.end

  if.then:
    ret i32 1

  if.end:
    ret i32 0
}

; We silently fixup cases where the register allocator or user has given us
; an instruction with incorrect operands that is trivially acceptable.
; beqc and bnec have the restriction that $rs < $rt.

define i32 @f2(i32 %a, i32 %b) {
; ENCODING-LABEL: f2
; ENCODING-NOT:   beqc $5, $4
; ENCODING-NOT:   bnec $5, $4

  %cmp = icmp eq i32 %b, %a
  br i1 %cmp, label %if.then, label %if.end

  if.then:
    ret i32 1

  if.end:
    ret i32 0
}

