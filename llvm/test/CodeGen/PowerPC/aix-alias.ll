; TODO: Add object generation test when visibility for object generation
;       is implemnted.

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -data-sections=false -xcoff-traceback-table=false < %s | \
; RUN:   FileCheck --check-prefix=ASM %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -data-sections=false -xcoff-traceback-table=false < %s | \
; RUN:   FileCheck --check-prefix=ASM %s

@var = global i32 42
@var1 = alias i32, ptr @var
@var2 = alias i32, ptr @var1
@var_l = linkonce_odr alias i32, ptr @var
@var_i = internal alias i32, ptr @var
@var_h = hidden alias i32, ptr @var
@var_p = protected alias i32, ptr @var

@array = global [2 x i32] [i32 1, i32 2], align 4
@x = global ptr getelementptr (i8, ptr @array, i64 4), align 4
@bitcast_alias = alias ptr, ptr @x

define i32 @fun() {
  ret i32 0
}

%FunTy = type i32()
@fun_weak = weak alias %FunTy, ptr @fun
@fun_hidden = hidden alias %FunTy, ptr @fun
@fun_ptr = global ptr @fun_weak

define i32 @test() {
entry:
   %tmp = load i32, ptr @var1
   %tmp1 = load i32, ptr @var2
   %tmp0 = load i32, ptr @var_i
   %tmp2 = call i32 @fun()
   %tmp3 = add i32 %tmp, %tmp2
   %tmp4 = call i32 @fun_weak()
   %tmp5 = add i32 %tmp3, %tmp4
   %tmp6 = add i32 %tmp1, %tmp5
   %tmp7 = add i32 %tmp6, %tmp0
   %fun_ptr1 = alloca ptr
   store ptr @fun_weak, ptr %fun_ptr1
   %tmp8 = call i32 %fun_ptr1()
   %tmp9 = call i32 @fun_hidden()
   %tmp10 = add i32 %tmp7, %tmp8
   %tmp11 = add i32 %tmp10, %tmp9
   ret i32 %tmp11
}

; ASM:         .globl  fun[DS]
; ASM-NEXT:    .globl  .fun
; ASM-NEXT:    .align  4
; ASM-NEXT:    .csect fun[DS]
; ASM-NEXT:  fun_weak:                               # @fun
; ASM-NEXT:  fun_hidden:
; ASM:         .csect .text[PR],5
; ASM-NEXT:  .fun:
; ASM-NEXT:  .fun_weak:
; ASM-NEXT:  .fun_hidden:
; ASM-NEXT:  # %bb.0:
; ASM-NEXT:    li 3, 0
; ASM-NEXT:    blr
; ASM-NEXT:                                          # -- End function
; ASM:       .csect .text[PR],5
; ASM-NEXT:  .test:
; ASM-NEXT:  # %bb.0:                                # %entry
; ASM:         bl .fun
; ASM-NEXT:    nop
; ASM:         bl .fun_weak
; ASM-NEXT:    nop
; ASM:         bl .fun_hidden
; ASM:                                               # -- End function
; ASM-NEXT:    .csect .data[RW]
; ASM-NEXT:    .globl  var
; ASM-NEXT:    .globl  var1
; ASM-NEXT:    .globl  var2
; ASM-NEXT:    .weak   var_l
; ASM-NEXT:    .lglobl var_i
; ASM-NEXT:    .globl  var_h,hidden
; ASM-NEXT:    .globl  var_p,protected
; ASM:       var:
; ASM-NEXT:  var1:
; ASM-NEXT:  var2:
; ASM-NEXT:  var_l:
; ASM-NEXT:  var_i:
; ASM-NEXT:  var_h:
; ASM-NEXT:  var_p:
; ASM-NEXT:    .vbyte  4, 42
; ASM-NEXT:    .globl array
; ASM:       array:
; ASM-NEXT:    .vbyte 4, 1 # 0x1
; ASM-NEXT:    .vbyte 4, 2 # 0x2
; ASM-NEXT:    .globl x
; ASM-NEXT:    .globl  bitcast_alias
; ASM:       x:
; ASM-NEXT:  bitcast_alias:
; ASM-NEXT:    .vbyte  {{[0-9]+}}, array+4
; ASM-NEXT:    .globl  fun_ptr
; ASM:       fun_ptr:
; ASM-NEXT:    .vbyte  {{[0-9]+}}, fun_weak
; ASM-NEXT:    .weak fun_weak
; ASM-NEXT:    .weak .fun_weak
; ASM-NEXT:    .globl  fun_hidden,hidden
; ASM-NEXT:    .globl  .fun_hidden,hidden
; ASM-NEXT:    .toc
; ASM-NEXT:  L..C0:
; ASM-NEXT:    .tc var1[TC],var1
; ASM-NEXT:  L..C1:
; ASM-NEXT:    .tc var2[TC],var2
; ASM-NEXT:  L..C2:
; ASM-NEXT:    .tc var_i[TC],var_i
; ASM-NEXT:  L..C3:
; ASM-NEXT:    .tc fun_weak[TC],fun_weak
