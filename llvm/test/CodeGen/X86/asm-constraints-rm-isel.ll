; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel=0 \
; RUN:   -stop-after=finalize-isel < %s | FileCheck --check-prefixes=CHECK,X64 %s
; RUN: llc -mtriple=i686-unknown-linux-gnu -global-isel=0 \
; RUN:   -stop-after=finalize-isel < %s | FileCheck --check-prefixes=CHECK,X86 %s
; The -O0 RUN line tests the optimization level, which keeps inputs in memory.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel=0 -O0 \
; RUN:   -stop-after=finalize-isel < %s | FileCheck --check-prefix=O0 %s

; An "rm" operand picks a register, marked foldable so that the register
; allocator can still put it in a stack slot, but only when the allocator can
; fold it: a direct integer or pointer operand that fills one register.
; Everything else keeps using memory.

%struct.S = type { [3 x i8] }

define void @input_i32(i32 %x) {
; CHECK-LABEL: name: input_i32
; CHECK: INLINEASM {{.*}}, reguse:GR32{{[_A-Z0-9]*}} foldable, %{{[0-9]+}}
; O0-LABEL: name: input_i32
; O0: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "rm"(i32 %x)
  ret void
}

; Two registers on i686, which can't be folded as one.
define void @input_i64(i64 %x) {
; CHECK-LABEL: name: input_i64
; X64: INLINEASM {{.*}}, reguse:GR64{{[_A-Z0-9]*}} foldable, %{{[0-9]+}}
; X86: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "rm"(i64 %x)
  ret void
}

; A floating-point value would have to move to a general-purpose register.
define void @input_float(float %x) {
; CHECK-LABEL: name: input_float
; CHECK: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "rm"(float %x)
  ret void
}

define void @input_double(double %x) {
; CHECK-LABEL: name: input_double
; CHECK: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "rm"(double %x)
  ret void
}

define void @input_i128(i128 %x) {
; CHECK-LABEL: name: input_i128
; CHECK: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "rm"(i128 %x)
  ret void
}

; 'r' would give these a GR64 (GR32 on i686), too wide for the value.
define void @input_half(half %x) {
; CHECK-LABEL: name: input_half
; CHECK: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "rm"(half %x)
  ret void
}

define void @input_bfloat(bfloat %x) {
; CHECK-LABEL: name: input_bfloat
; CHECK: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "rm"(bfloat %x)
  ret void
}

; No 'r' register can hold these at all.
define void @input_x86_fp80(x86_fp80 %x) {
; CHECK-LABEL: name: input_x86_fp80
; CHECK: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "rm"(x86_fp80 %x)
  ret void
}

define void @input_v4f32(<4 x float> %x) {
; CHECK-LABEL: name: input_v4f32
; CHECK: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "rm"(<4 x float> %x)
  ret void
}

; An indirect operand already names memory, as Clang emits for an "rm" input
; of aggregate type and for every "=rm" output.
define void @input_indirect(ptr %p) {
; CHECK-LABEL: name: input_indirect
; CHECK: INLINEASM {{.*}}, mem:m ops:5, {{(killed )?}}%0,
  call void asm sideeffect "# $0", "*rm"(ptr elementtype(%struct.S) %p)
  ret void
}

define void @output_indirect(ptr %p) {
; CHECK-LABEL: name: output_indirect
; CHECK: INLINEASM {{.*}}, mem:m ops:5, {{(killed )?}}%0,
  call void asm sideeffect "# $0", "=*rm"(ptr elementtype(i64) %p)
  ret void
}

; A direct output can't use memory, so it takes the register at -O0, too.
define i32 @output_i32() {
; CHECK-LABEL: name: output_i32
; CHECK: INLINEASM {{.*}}, regdef:GR32{{[_A-Z0-9]*}} foldable, def %{{[0-9]+}}
; O0-LABEL: name: output_i32
; O0: INLINEASM {{.*}}, regdef:GR32{{[_A-Z0-9]*}} foldable, def %{{[0-9]+}}
  %r = call i32 asm sideeffect "# $0", "=rm"()
  ret i32 %r
}

; A tied pair is folded through its output.
define i32 @inout_i32(i32 %x) {
; CHECK-LABEL: name: inout_i32
; CHECK: INLINEASM {{.*}}, regdef:GR32{{[_A-Z0-9]*}} foldable, def %{{[0-9]+}}, reguse tiedto:$0, %{{[0-9]+}}(tied-def 3)
  %r = call i32 asm sideeffect "# $0", "=rm,0"(i32 %x)
  ret i32 %r
}

; Clang's form of "+rm": an indirect output can't use memory when tied, so it
; was already a register, which stays unfoldable.
define void @inout_indirect_i32(ptr %p, i32 %x) {
; CHECK-LABEL: name: inout_indirect_i32
; CHECK: INLINEASM {{.*}}, regdef:GR32{{[_A-Z0-9]*}}, def %{{[0-9]+}}, reguse tiedto:$0, %{{[0-9]+}}(tied-def 3)
  call void asm sideeffect "# $0", "=*rm,0"(ptr elementtype(i32) %p, i32 %x)
  ret void
}

; Clang expands "g" to "imr". The immediate is still tried first, so only a
; non-constant value takes the foldable register.
define void @input_g_i32(i32 %x) {
; CHECK-LABEL: name: input_g_i32
; CHECK: INLINEASM {{.*}}, reguse:GR32{{[_A-Z0-9]*}} foldable, %{{[0-9]+}}
  call void asm sideeffect "# $0", "imr"(i32 %x)
  ret void
}

define void @input_g_const() {
; CHECK-LABEL: name: input_g_const
; CHECK: INLINEASM {{.*}}, imm, 42
  call void asm sideeffect "# $0", "imr"(i32 42)
  ret void
}

; The order of the codes doesn't matter.
define void @input_mri_i32(i32 %x) {
; CHECK-LABEL: name: input_mri_i32
; CHECK: INLINEASM {{.*}}, reguse:GR32{{[_A-Z0-9]*}} foldable, %{{[0-9]+}}
  call void asm sideeffect "# $0", "mri"(i32 %x)
  ret void
}

define void @input_g_i128(i128 %x) {
; CHECK-LABEL: name: input_g_i128
; CHECK: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "imr"(i128 %x)
  ret void
}

define void @input_g_indirect(ptr %p) {
; CHECK-LABEL: name: input_g_indirect
; CHECK: INLINEASM {{.*}}, mem:m ops:5, {{(killed )?}}%0,
  call void asm sideeffect "# $0", "*imr"(ptr elementtype(i32) %p)
  ret void
}

define i32 @output_g_i32() {
; CHECK-LABEL: name: output_g_i32
; CHECK: INLINEASM {{.*}}, regdef:GR32{{[_A-Z0-9]*}} foldable, def %{{[0-9]+}}
  %r = call i32 asm sideeffect "# $0", "=imr"()
  ret i32 %r
}

define i32 @inout_g_i32(i32 %x) {
; CHECK-LABEL: name: inout_g_i32
; CHECK: INLINEASM {{.*}}, regdef:GR32{{[_A-Z0-9]*}} foldable, def %{{[0-9]+}}, reguse tiedto:$0, %{{[0-9]+}}(tied-def 3)
  %r = call i32 asm sideeffect "# $0", "=imr,0"(i32 %x)
  ret i32 %r
}

; Another register class besides 'r' leaves the usual choice of memory.
define void @input_qmr_i32(i32 %x) {
; CHECK-LABEL: name: input_qmr_i32
; CHECK: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "qmr"(i32 %x)
  ret void
}

; Instruction selection treats an optnone function as -O0, which the
; register allocator that folds doesn't.
define void @input_i32_optnone(i32 %x) noinline optnone {
; CHECK-LABEL: name: input_i32_optnone
; CHECK: INLINEASM {{.*}}, mem:m ops:5, %stack.0,
  call void asm sideeffect "# $0", "rm"(i32 %x)
  ret void
}

define i32 @output_i32_optnone() noinline optnone {
; CHECK-LABEL: name: output_i32_optnone
; CHECK: INLINEASM {{.*}}, regdef:GR32{{[_A-Z0-9]*}} foldable, def %{{[0-9]+}}
  %r = call i32 asm sideeffect "# $0", "=rm"()
  ret i32 %r
}

; The greedy allocator can't fold an asm goto output tied to an input, so it
; stays a register as before. An untied one can be folded.
define i32 @callbr_inout_i32(i32 %x) {
; CHECK-LABEL: name: callbr_inout_i32
; CHECK: INLINEASM_BR {{.*}}, regdef:GR32{{[_A-Z0-9]*}}, def %{{[0-9]+}}, reguse tiedto:$0, %{{[0-9]+}}(tied-def 3)
  %r = callbr i32 asm sideeffect "# $0 $2", "=rm,0,!i"(i32 %x)
      to label %fallthrough [label %indirect]
fallthrough:
  ret i32 %r
indirect:
  ret i32 0
}

define i32 @callbr_output_i32() {
; CHECK-LABEL: name: callbr_output_i32
; CHECK: INLINEASM_BR {{.*}}, regdef:GR32{{[_A-Z0-9]*}} foldable, def %{{[0-9]+}}
  %r = callbr i32 asm sideeffect "# $0 $1", "=rm,!i"()
      to label %fallthrough [label %indirect]
fallthrough:
  ret i32 %r
indirect:
  ret i32 0
}
