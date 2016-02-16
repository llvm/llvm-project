; RUN: llc < %s -stack-symbol-ordering=0 -disable-fp-elim -mtriple=x86_64-pc-linux-gnu -mcpu=corei7 -o - | FileCheck %s
;  This test is fairly fragile.  The goal is to ensure that "large" stack
;  objects are allocated closest to the stack protector (i.e., farthest away 
;  from the Stack Pointer.)  In standard SSP mode this means that large (>=
;  ssp-buffer-size) arrays and structures containing such arrays are
;  closet to the protector.  With sspstrong and sspreq this means large
;  arrays/structures-with-arrays are closest, followed by small (< ssp-buffer-size)
;  arrays/structures-with-arrays, and then addr-taken variables.
;
;  Ideally, we only want verify that the objects appear in the correct groups
;  and that the groups have the correct relative stack offset.  The ordering
;  within a group is not relevant to this test.  Unfortunately, there is not
;  an elegant way to do this, so just match the offset for each object.
; RUN: llc < %s -disable-fp-elim -mtriple=x86_64-unknown-unknown -O0 -mcpu=corei7 -o - \
; RUN:   | FileCheck --check-prefix=FAST-NON-LIN %s
; FastISel was not setting the StackProtectorIndex when lowering
; Intrinsic::stackprotector and as a result the stack re-arrangement code was
; never applied.  This problem only shows up on non-Linux platforms because on
; Linux the stack protector cookie is loaded from a special address space which
; always triggers standard ISel.  Run a basic test to ensure that at -O0
; on a non-linux target the data layout rules are triggered.

%struct.struct_large_char = type { [8 x i8] }
%struct.struct_small_char = type { [2 x i8] }
%struct.struct_large_nonchar = type { [8 x i32] }
%struct.struct_small_nonchar = type { [2 x i16] }

define void @layout_ssp() ssp {
entry:
; Expected stack layout for ssp is
;  -16 large_char          . Group 1, nested arrays, arrays >= ssp-buffer-size
;  -24 struct_large_char   .
;  -28 scalar1             | Everything else
;  -32 scalar2
;  -36 scalar3
;  -40 addr-of
;  -44 small_nonchar
;  -80 large_nonchar
;  -82 small_char
;  -88 struct_small_char
;  -120 struct_large_nonchar
;  -128 struct_small_nonchar

; CHECK: layout_ssp:
; CHECK: call{{l|q}} get_scalar1
; CHECK: movl %eax, -28(
; CHECK: call{{l|q}} end_scalar1

; CHECK: call{{l|q}} get_scalar2
; CHECK: movl %eax, -32(
; CHECK: call{{l|q}} end_scalar2

; CHECK: call{{l|q}} get_scalar3
; CHECK: movl %eax, -36(
; CHECK: call{{l|q}} end_scalar3

; CHECK: call{{l|q}} get_addrof
; CHECK: movl %eax, -40(
; CHECK: call{{l|q}} end_addrof

; CHECK: get_small_nonchar
; CHECK: movw %ax, -44(
; CHECK: call{{l|q}} end_small_nonchar

; CHECK: call{{l|q}} get_large_nonchar
; CHECK: movl %eax, -80(
; CHECK: call{{l|q}} end_large_nonchar

; CHECK: call{{l|q}} get_small_char
; CHECK: movb %al, -82(
; CHECK: call{{l|q}} end_small_char

; CHECK: call{{l|q}} get_large_char
; CHECK: movb %al, -16(
; CHECK: call{{l|q}} end_large_char

; CHECK: call{{l|q}} get_struct_large_char
; CHECK: movb %al, -24(
; CHECK: call{{l|q}} end_struct_large_char

; CHECK: call{{l|q}} get_struct_small_char
; CHECK: movb %al, -88(
; CHECK: call{{l|q}} end_struct_small_char

; CHECK: call{{l|q}} get_struct_large_nonchar
; CHECK: movl %eax, -120(
; CHECK: call{{l|q}} end_struct_large_nonchar

; CHECK: call{{l|q}} get_struct_small_nonchar
; CHECK: movw %ax, -128(
; CHECK: call{{l|q}} end_struct_small_nonchar
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %ptr = alloca i32, align 4
  %small2 = alloca [2 x i16], align 2
  %large2 = alloca [8 x i32], align 16
  %small = alloca [2 x i8], align 1
  %large = alloca [8 x i8], align 1
  %a = alloca %struct.struct_large_char, align 1
  %b = alloca %struct.struct_small_char, align 1
  %c = alloca %struct.struct_large_nonchar, align 8
  %d = alloca %struct.struct_small_nonchar, align 2
  %call = call i32 @get_scalar1()
  store i32 %call, i32* %x, align 4
  call void @end_scalar1()
  %call1 = call i32 @get_scalar2()
  store i32 %call1, i32* %y, align 4
  call void @end_scalar2()
  %call2 = call i32 @get_scalar3()
  store i32 %call2, i32* %z, align 4
  call void @end_scalar3()
  %call3 = call i32 @get_addrof()
  store i32 %call3, i32* %ptr, align 4
  call void @end_addrof()
  %call4 = call signext i16 @get_small_nonchar()
  %arrayidx = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i64 0
  store i16 %call4, i16* %arrayidx, align 2
  call void @end_small_nonchar()
  %call5 = call i32 @get_large_nonchar()
  %arrayidx6 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i64 0
  store i32 %call5, i32* %arrayidx6, align 4
  call void @end_large_nonchar()
  %call7 = call signext i8 @get_small_char()
  %arrayidx8 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i64 0
  store i8 %call7, i8* %arrayidx8, align 1
  call void @end_small_char()
  %call9 = call signext i8 @get_large_char()
  %arrayidx10 = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i64 0
  store i8 %call9, i8* %arrayidx10, align 1
  call void @end_large_char()
  %call11 = call signext i8 @get_struct_large_char()
  %foo = getelementptr inbounds %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %arrayidx12 = getelementptr inbounds [8 x i8], [8 x i8]* %foo, i32 0, i64 0
  store i8 %call11, i8* %arrayidx12, align 1
  call void @end_struct_large_char()
  %call13 = call signext i8 @get_struct_small_char()
  %foo14 = getelementptr inbounds %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %arrayidx15 = getelementptr inbounds [2 x i8], [2 x i8]* %foo14, i32 0, i64 0
  store i8 %call13, i8* %arrayidx15, align 1
  call void @end_struct_small_char()
  %call16 = call i32 @get_struct_large_nonchar()
  %foo17 = getelementptr inbounds %struct.struct_large_nonchar, %struct.struct_large_nonchar* %c, i32 0, i32 0
  %arrayidx18 = getelementptr inbounds [8 x i32], [8 x i32]* %foo17, i32 0, i64 0
  store i32 %call16, i32* %arrayidx18, align 4
  call void @end_struct_large_nonchar()
  %call19 = call signext i16 @get_struct_small_nonchar()
  %foo20 = getelementptr inbounds %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %arrayidx21 = getelementptr inbounds [2 x i16], [2 x i16]* %foo20, i32 0, i64 0
  store i16 %call19, i16* %arrayidx21, align 2
  call void @end_struct_small_nonchar()
  %arraydecay = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i32 0
  %arraydecay22 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i32 0
  %arraydecay23 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i32 0
  %arraydecay24 = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i32 0
  %0 = load i32, i32* %x, align 4
  %1 = load i32, i32* %y, align 4
  %2 = load i32, i32* %z, align 4
  %coerce.dive = getelementptr %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %3 = bitcast [8 x i8]* %coerce.dive to i64*
  %4 = load i64, i64* %3, align 1
  %coerce.dive25 = getelementptr %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %5 = bitcast [2 x i8]* %coerce.dive25 to i16*
  %6 = load i16, i16* %5, align 1
  %coerce.dive26 = getelementptr %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %7 = bitcast [2 x i16]* %coerce.dive26 to i32*
  %8 = load i32, i32* %7, align 1
  call void @takes_all(i64 %4, i16 %6, %struct.struct_large_nonchar* byval align 8 %c, i32 %8, i8* %arraydecay, i8* %arraydecay22, i32* %arraydecay23, i16* %arraydecay24, i32* %ptr, i32 %0, i32 %1, i32 %2)
  ret void
}

define void @layout_sspstrong() nounwind uwtable sspstrong {
entry:
; Expected stack layout for sspstrong is
;   -48   large_nonchar          . Group 1, nested arrays,
;   -56   large_char             .  arrays >= ssp-buffer-size
;   -64   struct_large_char      .
;   -96   struct_large_nonchar   .
;   -100  small_non_char         | Group 2, nested arrays, 
;   -102  small_char             |  arrays < ssp-buffer-size
;   -104  struct_small_char      |
;   -112  struct_small_nonchar   |
;   -116  addrof                 * Group 3, addr-of local
;   -120  scalar                 + Group 4, everything else
;   -124  scalar                 +
;   -128  scalar                 +
;   
; CHECK: layout_sspstrong:
; CHECK: call{{l|q}} get_scalar1
; CHECK: movl %eax, -120(
; CHECK: call{{l|q}} end_scalar1

; CHECK: call{{l|q}} get_scalar2
; CHECK: movl %eax, -124(
; CHECK: call{{l|q}} end_scalar2

; CHECK: call{{l|q}} get_scalar3
; CHECK: movl %eax, -128(
; CHECK: call{{l|q}} end_scalar3

; CHECK: call{{l|q}} get_addrof
; CHECK: movl %eax, -116(
; CHECK: call{{l|q}} end_addrof

; CHECK: get_small_nonchar
; CHECK: movw %ax, -100(
; CHECK: call{{l|q}} end_small_nonchar

; CHECK: call{{l|q}} get_large_nonchar
; CHECK: movl %eax, -48(
; CHECK: call{{l|q}} end_large_nonchar

; CHECK: call{{l|q}} get_small_char
; CHECK: movb %al, -102(
; CHECK: call{{l|q}} end_small_char

; CHECK: call{{l|q}} get_large_char
; CHECK: movb %al, -56(
; CHECK: call{{l|q}} end_large_char

; CHECK: call{{l|q}} get_struct_large_char
; CHECK: movb %al, -64(
; CHECK: call{{l|q}} end_struct_large_char

; CHECK: call{{l|q}} get_struct_small_char
; CHECK: movb %al, -104(
; CHECK: call{{l|q}} end_struct_small_char

; CHECK: call{{l|q}} get_struct_large_nonchar
; CHECK: movl %eax, -96(
; CHECK: call{{l|q}} end_struct_large_nonchar

; CHECK: call{{l|q}} get_struct_small_nonchar
; CHECK: movw %ax, -112(
; CHECK: call{{l|q}} end_struct_small_nonchar
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %ptr = alloca i32, align 4
  %small2 = alloca [2 x i16], align 2
  %large2 = alloca [8 x i32], align 16
  %small = alloca [2 x i8], align 1
  %large = alloca [8 x i8], align 1
  %a = alloca %struct.struct_large_char, align 1
  %b = alloca %struct.struct_small_char, align 1
  %c = alloca %struct.struct_large_nonchar, align 8
  %d = alloca %struct.struct_small_nonchar, align 2
  %call = call i32 @get_scalar1()
  store i32 %call, i32* %x, align 4
  call void @end_scalar1()
  %call1 = call i32 @get_scalar2()
  store i32 %call1, i32* %y, align 4
  call void @end_scalar2()
  %call2 = call i32 @get_scalar3()
  store i32 %call2, i32* %z, align 4
  call void @end_scalar3()
  %call3 = call i32 @get_addrof()
  store i32 %call3, i32* %ptr, align 4
  call void @end_addrof()
  %call4 = call signext i16 @get_small_nonchar()
  %arrayidx = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i64 0
  store i16 %call4, i16* %arrayidx, align 2
  call void @end_small_nonchar()
  %call5 = call i32 @get_large_nonchar()
  %arrayidx6 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i64 0
  store i32 %call5, i32* %arrayidx6, align 4
  call void @end_large_nonchar()
  %call7 = call signext i8 @get_small_char()
  %arrayidx8 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i64 0
  store i8 %call7, i8* %arrayidx8, align 1
  call void @end_small_char()
  %call9 = call signext i8 @get_large_char()
  %arrayidx10 = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i64 0
  store i8 %call9, i8* %arrayidx10, align 1
  call void @end_large_char()
  %call11 = call signext i8 @get_struct_large_char()
  %foo = getelementptr inbounds %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %arrayidx12 = getelementptr inbounds [8 x i8], [8 x i8]* %foo, i32 0, i64 0
  store i8 %call11, i8* %arrayidx12, align 1
  call void @end_struct_large_char()
  %call13 = call signext i8 @get_struct_small_char()
  %foo14 = getelementptr inbounds %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %arrayidx15 = getelementptr inbounds [2 x i8], [2 x i8]* %foo14, i32 0, i64 0
  store i8 %call13, i8* %arrayidx15, align 1
  call void @end_struct_small_char()
  %call16 = call i32 @get_struct_large_nonchar()
  %foo17 = getelementptr inbounds %struct.struct_large_nonchar, %struct.struct_large_nonchar* %c, i32 0, i32 0
  %arrayidx18 = getelementptr inbounds [8 x i32], [8 x i32]* %foo17, i32 0, i64 0
  store i32 %call16, i32* %arrayidx18, align 4
  call void @end_struct_large_nonchar()
  %call19 = call signext i16 @get_struct_small_nonchar()
  %foo20 = getelementptr inbounds %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %arrayidx21 = getelementptr inbounds [2 x i16], [2 x i16]* %foo20, i32 0, i64 0
  store i16 %call19, i16* %arrayidx21, align 2
  call void @end_struct_small_nonchar()
  %arraydecay = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i32 0
  %arraydecay22 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i32 0
  %arraydecay23 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i32 0
  %arraydecay24 = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i32 0
  %0 = load i32, i32* %x, align 4
  %1 = load i32, i32* %y, align 4
  %2 = load i32, i32* %z, align 4
  %coerce.dive = getelementptr %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %3 = bitcast [8 x i8]* %coerce.dive to i64*
  %4 = load i64, i64* %3, align 1
  %coerce.dive25 = getelementptr %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %5 = bitcast [2 x i8]* %coerce.dive25 to i16*
  %6 = load i16, i16* %5, align 1
  %coerce.dive26 = getelementptr %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %7 = bitcast [2 x i16]* %coerce.dive26 to i32*
  %8 = load i32, i32* %7, align 1
  call void @takes_all(i64 %4, i16 %6, %struct.struct_large_nonchar* byval align 8 %c, i32 %8, i8* %arraydecay, i8* %arraydecay22, i32* %arraydecay23, i16* %arraydecay24, i32* %ptr, i32 %0, i32 %1, i32 %2)
  ret void
}

define void @layout_sspreq() nounwind uwtable sspreq {
entry:
; Expected stack layout for sspreq is the same as sspstrong
;   
; CHECK: layout_sspreq:
; CHECK: call{{l|q}} get_scalar1
; CHECK: movl %eax, -120(
; CHECK: call{{l|q}} end_scalar1

; CHECK: call{{l|q}} get_scalar2
; CHECK: movl %eax, -124(
; CHECK: call{{l|q}} end_scalar2

; CHECK: call{{l|q}} get_scalar3
; CHECK: movl %eax, -128(
; CHECK: call{{l|q}} end_scalar3

; CHECK: call{{l|q}} get_addrof
; CHECK: movl %eax, -116(
; CHECK: call{{l|q}} end_addrof

; CHECK: get_small_nonchar
; CHECK: movw %ax, -100(
; CHECK: call{{l|q}} end_small_nonchar

; CHECK: call{{l|q}} get_large_nonchar
; CHECK: movl %eax, -48(
; CHECK: call{{l|q}} end_large_nonchar

; CHECK: call{{l|q}} get_small_char
; CHECK: movb %al, -102(
; CHECK: call{{l|q}} end_small_char

; CHECK: call{{l|q}} get_large_char
; CHECK: movb %al, -56(
; CHECK: call{{l|q}} end_large_char

; CHECK: call{{l|q}} get_struct_large_char
; CHECK: movb %al, -64(
; CHECK: call{{l|q}} end_struct_large_char

; CHECK: call{{l|q}} get_struct_small_char
; CHECK: movb %al, -104(
; CHECK: call{{l|q}} end_struct_small_char

; CHECK: call{{l|q}} get_struct_large_nonchar
; CHECK: movl %eax, -96(
; CHECK: call{{l|q}} end_struct_large_nonchar

; CHECK: call{{l|q}} get_struct_small_nonchar
; CHECK: movw %ax, -112(
; CHECK: call{{l|q}} end_struct_small_nonchar
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %ptr = alloca i32, align 4
  %small2 = alloca [2 x i16], align 2
  %large2 = alloca [8 x i32], align 16
  %small = alloca [2 x i8], align 1
  %large = alloca [8 x i8], align 1
  %a = alloca %struct.struct_large_char, align 1
  %b = alloca %struct.struct_small_char, align 1
  %c = alloca %struct.struct_large_nonchar, align 8
  %d = alloca %struct.struct_small_nonchar, align 2
  %call = call i32 @get_scalar1()
  store i32 %call, i32* %x, align 4
  call void @end_scalar1()
  %call1 = call i32 @get_scalar2()
  store i32 %call1, i32* %y, align 4
  call void @end_scalar2()
  %call2 = call i32 @get_scalar3()
  store i32 %call2, i32* %z, align 4
  call void @end_scalar3()
  %call3 = call i32 @get_addrof()
  store i32 %call3, i32* %ptr, align 4
  call void @end_addrof()
  %call4 = call signext i16 @get_small_nonchar()
  %arrayidx = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i64 0
  store i16 %call4, i16* %arrayidx, align 2
  call void @end_small_nonchar()
  %call5 = call i32 @get_large_nonchar()
  %arrayidx6 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i64 0
  store i32 %call5, i32* %arrayidx6, align 4
  call void @end_large_nonchar()
  %call7 = call signext i8 @get_small_char()
  %arrayidx8 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i64 0
  store i8 %call7, i8* %arrayidx8, align 1
  call void @end_small_char()
  %call9 = call signext i8 @get_large_char()
  %arrayidx10 = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i64 0
  store i8 %call9, i8* %arrayidx10, align 1
  call void @end_large_char()
  %call11 = call signext i8 @get_struct_large_char()
  %foo = getelementptr inbounds %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %arrayidx12 = getelementptr inbounds [8 x i8], [8 x i8]* %foo, i32 0, i64 0
  store i8 %call11, i8* %arrayidx12, align 1
  call void @end_struct_large_char()
  %call13 = call signext i8 @get_struct_small_char()
  %foo14 = getelementptr inbounds %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %arrayidx15 = getelementptr inbounds [2 x i8], [2 x i8]* %foo14, i32 0, i64 0
  store i8 %call13, i8* %arrayidx15, align 1
  call void @end_struct_small_char()
  %call16 = call i32 @get_struct_large_nonchar()
  %foo17 = getelementptr inbounds %struct.struct_large_nonchar, %struct.struct_large_nonchar* %c, i32 0, i32 0
  %arrayidx18 = getelementptr inbounds [8 x i32], [8 x i32]* %foo17, i32 0, i64 0
  store i32 %call16, i32* %arrayidx18, align 4
  call void @end_struct_large_nonchar()
  %call19 = call signext i16 @get_struct_small_nonchar()
  %foo20 = getelementptr inbounds %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %arrayidx21 = getelementptr inbounds [2 x i16], [2 x i16]* %foo20, i32 0, i64 0
  store i16 %call19, i16* %arrayidx21, align 2
  call void @end_struct_small_nonchar()
  %arraydecay = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i32 0
  %arraydecay22 = getelementptr inbounds [2 x i8], [2 x i8]* %small, i32 0, i32 0
  %arraydecay23 = getelementptr inbounds [8 x i32], [8 x i32]* %large2, i32 0, i32 0
  %arraydecay24 = getelementptr inbounds [2 x i16], [2 x i16]* %small2, i32 0, i32 0
  %0 = load i32, i32* %x, align 4
  %1 = load i32, i32* %y, align 4
  %2 = load i32, i32* %z, align 4
  %coerce.dive = getelementptr %struct.struct_large_char, %struct.struct_large_char* %a, i32 0, i32 0
  %3 = bitcast [8 x i8]* %coerce.dive to i64*
  %4 = load i64, i64* %3, align 1
  %coerce.dive25 = getelementptr %struct.struct_small_char, %struct.struct_small_char* %b, i32 0, i32 0
  %5 = bitcast [2 x i8]* %coerce.dive25 to i16*
  %6 = load i16, i16* %5, align 1
  %coerce.dive26 = getelementptr %struct.struct_small_nonchar, %struct.struct_small_nonchar* %d, i32 0, i32 0
  %7 = bitcast [2 x i16]* %coerce.dive26 to i32*
  %8 = load i32, i32* %7, align 1
  call void @takes_all(i64 %4, i16 %6, %struct.struct_large_nonchar* byval align 8 %c, i32 %8, i8* %arraydecay, i8* %arraydecay22, i32* %arraydecay23, i16* %arraydecay24, i32* %ptr, i32 %0, i32 %1, i32 %2)
  ret void
}

define void @fast_non_linux() ssp {
entry:
; FAST-NON-LIN: fast_non_linux:
; FAST-NON-LIN: call{{l|q}} get_scalar1
; FAST-NON-LIN: movl %eax, -20(
; FAST-NON-LIN: call{{l|q}} end_scalar1

; FAST-NON-LIN: call{{l|q}} get_large_char
; FAST-NON-LIN: movb %al, -16(
; FAST-NON-LIN: call{{l|q}} end_large_char
  %x = alloca i32, align 4
  %large = alloca [8 x i8], align 1
  %call = call i32 @get_scalar1()
  store i32 %call, i32* %x, align 4
  call void @end_scalar1()
  %call1 = call signext i8 @get_large_char()
  %arrayidx = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i64 0
  store i8 %call1, i8* %arrayidx, align 1
  call void @end_large_char()
  %0 = load i32, i32* %x, align 4
  %arraydecay = getelementptr inbounds [8 x i8], [8 x i8]* %large, i32 0, i32 0
  call void @takes_two(i32 %0, i8* %arraydecay)
  ret void
}

declare i32 @get_scalar1()
declare void @end_scalar1()

declare i32 @get_scalar2()
declare void @end_scalar2()

declare i32 @get_scalar3()
declare void @end_scalar3()

declare i32 @get_addrof()
declare void @end_addrof()

declare signext i16 @get_small_nonchar()
declare void @end_small_nonchar()

declare i32 @get_large_nonchar()
declare void @end_large_nonchar()

declare signext i8 @get_small_char()
declare void @end_small_char()

declare signext i8 @get_large_char()
declare void @end_large_char()

declare signext i8 @get_struct_large_char()
declare void @end_struct_large_char()

declare signext i8 @get_struct_small_char()
declare void @end_struct_small_char()

declare i32 @get_struct_large_nonchar()
declare void @end_struct_large_nonchar()

declare signext i16 @get_struct_small_nonchar()
declare void @end_struct_small_nonchar()

declare void @takes_all(i64, i16, %struct.struct_large_nonchar* byval align 8, i32, i8*, i8*, i32*, i16*, i32*, i32, i32, i32)
declare void @takes_two(i32, i8*)
