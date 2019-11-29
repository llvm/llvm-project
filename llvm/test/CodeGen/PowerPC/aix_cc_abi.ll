; RUN: llc -mtriple powerpc-ibm-aix-xcoff -stop-after=machine-cp -verify-machineinstrs < %s | \
; RUN: FileCheck --check-prefixes=CHECK,32BIT %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp -verify-machineinstrs < %s | \
; RUN: FileCheck --check-prefixes=CHECK,64BIT %s

define void @call_test_chars() {
entry:
  call i8 @test_chars(i8 signext 97, i8 signext 97, i8 signext 97, i8 signext 97)
  ret void
}

; CHECK-LABEL: name: call_test_chars

; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 97
; 32BIT: $r4 = LI 97
; 32BIT: $r5 = LI 97
; 32BIT: $r6 = LI 97
; 32BIT: BL_NOP <mcsymbol .test_chars>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit killed $r4, implicit killed $r5, implicit killed $r6, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 97
; 64BIT: $x4 = LI8 97
; 64BIT: $x5 = LI8 97
; 64BIT: $x6 = LI8 97
; 64BIT: BL8_NOP <mcsymbol .test_chars>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit killed $x4, implicit killed $x5, implicit killed $x6, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

define signext i8 @test_chars(i8 signext %c1, i8 signext %c2, i8 signext %c3, i8 signext %c4) {
entry:
  %conv = sext i8 %c1 to i32
  %conv1 = sext i8 %c2 to i32
  %add = add nsw i32 %conv, %conv1
  %conv2 = sext i8 %c3 to i32
  %add3 = add nsw i32 %add, %conv2
  %conv4 = sext i8 %c4 to i32
  %add5 = add nsw i32 %add3, %conv4
  %conv6 = trunc i32 %add5 to i8
  ret i8 %conv6
}

; CHECK-LABEL: name: test_chars

; 32BIT:       liveins:
; 32BIT-NEXT:  - { reg: '$r3', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r4', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r5', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r6', virtual-reg: '' }
; 32BIT:       body:
; 32BIT-NEXT:    bb.0.entry:
; 32BIT-NEXT:      liveins: $r3, $r4, $r5, $r6
 
; 64BIT:       liveins:
; 64BIT-NEXT:  - { reg: '$x3', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x4', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x5', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x6', virtual-reg: '' }
; 64BIT:       body:
; 64BIT-NEXT:    bb.0.entry:
; 64BIT-NEXT:      liveins: $x3, $x4, $x5, $x6

define void @call_test_chars_mix() {
entry:
  call i8 @test_chars_mix(i8 signext 97, i8 zeroext -31, i8 zeroext 97, i8 signext -31)
  ret void
}

; CHECK-LABEL: name: call_test_chars_mix

; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 97
; 32BIT: $r4 = LI 225
; 32BIT: $r5 = LI 97
; 32BIT: $r6 = LI -31
; 32BIT: BL_NOP <mcsymbol .test_chars_mix>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit killed $r4, implicit killed $r5, implicit killed $r6, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 97
; 64BIT: $x4 = LI8 225
; 64BIT: $x5 = LI8 97
; 64BIT: $x6 = LI8 -31
; 64BIT: BL8_NOP <mcsymbol .test_chars_mix>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit killed $x4, implicit killed $x5, implicit killed $x6, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

define signext i8 @test_chars_mix(i8 signext %c1, i8 zeroext %c2, i8 zeroext %c3, i8 signext %c4) {
entry:
  %conv = sext i8 %c1 to i32
  %conv1 = zext i8 %c2 to i32
  %add = add nsw i32 %conv, %conv1
  %conv2 = zext i8 %c3 to i32
  %add3 = add nsw i32 %add, %conv2
  %conv4 = sext i8 %c4 to i32
  %add5 = add nsw i32 %add3, %conv4
  %conv6 = trunc i32 %add5 to i8
  ret i8 %conv6
}

; CHECK-LABEL: name: test_chars_mix

; 32BIT:       liveins:
; 32BIT-NEXT:  - { reg: '$r3', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r4', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r5', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r6', virtual-reg: '' }
; 32BIT:       body:
; 32BIT-NEXT:    bb.0.entry:
; 32BIT-NEXT:      liveins: $r3, $r4, $r5, $r6
 
; 64BIT:       liveins:
; 64BIT-NEXT:  - { reg: '$x3', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x4', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x5', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x6', virtual-reg: '' }
; 64BIT:       body:
; 64BIT-NEXT:    bb.0.entry:
; 64BIT-NEXT:      liveins: $x3, $x4, $x5, $x6

@global_i1 = global i8 0, align 1

define  void @test_i1(i1 %b)  {
  entry:
   %frombool = zext i1 %b to i8
   store i8 %frombool, i8* @global_i1, align 1
   ret void
}

; 32BIT:       liveins:
; 32BIT-NEXT:  - { reg: '$r3', virtual-reg: '' }
; 32BIT:       body:             |
; 32BIT-NEXT:    bb.0.entry:
; 32BIT-NEXT:      liveins: $r3
; 32BIT:           renamable $r3 = RLWINM killed renamable $r3, 0, 31, 31
; 32BIT-NEXT:      STB killed renamable $r3, 0, killed renamable $r4 :: (store 1 into @global_i1)

; 64BIT:       liveins:
; 64BIT-NEXT:  - { reg: '$x3', virtual-reg: '' }
; 64BIT:       body:             |
; 64BIT-NEXT:    bb.0.entry:
; 64BIT-NEXT:      liveins: $x3
; 64BIT:           renamable $r[[REG1:[0-9]+]] = RLWINM renamable $r[[REG1]], 0, 31, 31, implicit killed $x3
; 64BIT-NEXT:      STB killed renamable $r[[REG1]], 0, killed renamable $x4 :: (store 1 into @global_i1)

define void @call_test_i1() {
entry:
  call void @test_i1(i1 1)
  ret void
}

; CHECK-LABEL: name: call_test_i1

; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 1
; 32BIT: BL_NOP <mcsymbol .test_i1>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 1
; 64BIT: BL8_NOP <mcsymbol .test_i1>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

define void @test_i1zext(i1 zeroext %b) {
  entry:
    %frombool = zext i1 %b to i8 
    store i8 %frombool, i8 * @global_i1, align 1 
    ret void 
  }

; 32BIT:       liveins:
; 32BIT-NEXT:  - { reg: '$r3', virtual-reg: '' }
; 32BIT:       body:             |
; 32BIT-NEXT:    bb.0.entry:
; 32BIT-NEXT:      liveins: $r3
; CHECK-NOT:       RLWINM 
; 32BIT:           STB killed renamable $r3, 0, killed renamable $r4 :: (store 1 into @global_i1)

; 64BIT:       liveins:
; 64BIT-NEXT:  - { reg: '$x3', virtual-reg: '' }
; 64BIT:       body:             |
; 64BIT-NEXT:    bb.0.entry:
; 64BIT-NEXT:      liveins: $x3
; CHECK-NOT:       RLWINM
; 64BIT:           STB8 killed renamable $x3, 0, killed renamable $x4 :: (store 1 into @global_i1)

define i32 @test_ints(i32 signext %a, i32 zeroext %b, i32 zeroext %c, i32 signext %d, i32 signext %e, i32 signext %f, i32 signext %g, i32 signext %h) {
entry:
    %add = add i32 %a, %b
    %add1 = add i32 %add, %c
    %add2 = add i32 %add1, %d
    %add3 = add i32 %add2, %e
    %add4 = add i32 %add3, %f
    %add5 = add i32 %add4, %g
    %add6 = add i32 %add5, %h
    ret i32 %add6
}

; CHECK-LABEL: name: test_ints

; 32BIT:       liveins:
; 32BIT-NEXT:  - { reg: '$r3', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r4', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r5', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r6', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r7', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r8', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r9', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r10', virtual-reg: '' }
; 32BIT:       body:             |
; 32BIT-NEXT:    bb.0.entry:
; 32BIT-NEXT:      liveins: $r3, $r4, $r5, $r6, $r7, $r8, $r9, $r10

; 64BIT:       liveins:
; 64BIT-NEXT:  - { reg: '$x3', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x4', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x5', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x6', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x7', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x8', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x9', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x10', virtual-reg: '' }
; 64BIT:       body:             |
; 64BIT-NEXT:    bb.0.entry:
; 64BIT-NEXT:      liveins: $x3, $x4, $x5, $x6, $x7, $x8, $x9, $x10

define void @call_test_ints() {
entry:
  call i32 @test_ints(i32 signext 1, i32 zeroext 1, i32 zeroext 2147483648, i32 signext -2147483648, i32 signext 1, i32 signext 1, i32 signext 1, i32 signext 1)
  ret void
}

; CHECK-LABEL: name: call_test_ints

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: renamable $x3 = LI8 1
; 64BIT: renamable $x5 = RLDICR killed renamable $x3, 31, 32
; 64BIT: $x3 = LI8 1
; 64BIT: $x4 = LI8 1
; 64BIT: $x6 = LIS8 32768
; 64BIT: $x7 = LI8 1
; 64BIT: $x8 = LI8 1
; 64BIT: $x9 = LI8 1
; 64BIT: $x10 = LI8 1
; 64BIT:  BL8_NOP <mcsymbol .test_ints>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit killed $x4, implicit $x5, implicit killed $x6, implicit killed $x7, implicit killed $x8, implicit killed $x9, implicit killed $x10, implicit $x2, implicit-def $r1, implicit-def dead $x3
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

define void @call_test_i64() {
entry:
  call i64 @test_i64(i64 1, i64 2, i64 3, i64 4)
  ret void
}


; CHECK-LABEL: name: call_test_i64

; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: $r3 = LI 0
; 32BIT: $r4 = LI 1
; 32BIT: $r5 = LI 0
; 32BIT: $r6 = LI 2
; 32BIT: $r7 = LI 0
; 32BIT: $r8 = LI 3
; 32BIT: $r9 = LI 0
; 32BIT: $r10 = LI 4
; 32BIT: BL_NOP <mcsymbol .test_i64>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit killed $r3, implicit killed $r4, implicit killed $r5, implicit killed $r6, implicit killed $r7, implicit killed $r8, implicit killed $r9, implicit killed $r10, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: $x3 = LI8 1
; 64BIT: $x4 = LI8 2
; 64BIT: $x5 = LI8 3
; 64BIT: $x6 = LI8 4
; 64BIT: BL8_NOP <mcsymbol .test_i64>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit killed $x3, implicit killed $x4, implicit killed $x5, implicit killed $x6, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

define i64 @test_i64(i64 %a, i64 %b, i64 %c, i64 %d) {
entry:
  %add = add nsw i64 %a, %b
  %add1 = add nsw i64 %add, %c
  %add2 = add nsw i64 %add1, %d
  ret i64 %add2
}

; CHECK-LABEL: name: test_i64

; 32BIT:       liveins:
; 32BIT-NEXT:  - { reg: '$r3', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r4', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r5', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r6', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r7', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r8', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r9', virtual-reg: '' }
; 32BIT-NEXT:  - { reg: '$r10', virtual-reg: '' }
; 32BIT:       body:             |
; 32BIT-NEXT:    bb.0.entry:
; 32BIT-NEXT:      liveins: $r3, $r4, $r5, $r6, $r7, $r8, $r9, $r10

; 64BIT:       liveins:
; 64BIT-NEXT:  - { reg: '$x3', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x4', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x5', virtual-reg: '' }
; 64BIT-NEXT:  - { reg: '$x6', virtual-reg: '' }
; 64BIT:       body:             |
; 64BIT-NEXT:    bb.0.entry:
; 64BIT-NEXT:      liveins: $x3, $x4, $x5, $x6

define void @call_test_int_ptr() {
entry:
  %b = alloca i32, align 4
  store i32 0, i32* %b, align 4
  call void @test_int_ptr(i32* %b)
  ret void
}

; CHECK-LABEL: name: call_test_int_ptr

; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: renamable $r3 = ADDI %stack.0.b, 0
; 32BIT: BL_NOP <mcsymbol .test_int_ptr>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: renamable $x3 = ADDI8 %stack.0.b, 0
; 64BIT: BL8_NOP <mcsymbol .test_int_ptr>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

define void @test_int_ptr(i32* %a) {
entry:
  %a.addr = alloca i32*, align 8
  store i32* %a, i32** %a.addr, align 8
  ret void
}

; CHECK-LABEL: name: test_int_ptr

; 32BIT:       liveins:
; 32BIT-NEXT:  - { reg: '$r3', virtual-reg: '' }
; 32BIT:       body:             |
; 32BIT-NEXT:    bb.0.entry:
; 32BIT-NEXT:      liveins: $r3
; 32BIT:           STW killed renamable $r3, 0, %stack.0.a.addr :: (store 4 into %ir.a.addr, align 8)

; 64BIT:       liveins:
; 64BIT-NEXT:  - { reg: '$x3', virtual-reg: '' }
; 64BIT:       body:             |
; 64BIT-NEXT:    bb.0.entry:
; 64BIT-NEXT:      liveins: $x3
; 64BIT:           STD killed renamable $x3, 0, %stack.0.a.addr :: (store 8 into %ir.a.addr)


define i32 @caller(i32 %i)  {
entry:
  %i.addr = alloca i32, align 4
  %b = alloca i8, align 1
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4
  %cmp = icmp ne i32 %0, 0
  %frombool = zext i1 %cmp to i8
  store i8 %frombool, i8* %b, align 1
  %1 = load i8, i8* %b, align 1
  %tobool = trunc i8 %1 to i1
  %call = call i32 @call_test_bool(i1 zeroext %tobool)
  ret i32 %call
}

declare i32 @call_test_bool(i1 zeroext)
  
; CHECK-LABEL: name:            caller

; 32BIT:        liveins:
; 32BIT-NEXT:   - { reg: '$r3', virtual-reg: '' }
; 32BIT:        body:             |
; 32BIT-NEXT:   bb.0.entry:
; 32BIT:         liveins: $r3
; 32BIT:          ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT:          BL_NOP <mcsymbol .call_test_bool>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r3, implicit $r2, implicit-def $r1, implicit-def $r3
; 32BIT:          ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT:        liveins:
; 64BIT-NEXT:   - { reg: '$x3', virtual-reg: '' }
; 64BIT:        body:             |
; 64BIT-NEXT:    bb.0.entry:
; 64BIT-NEXT:     liveins: $x3
; 64BIT:          ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT:          BL8_NOP <mcsymbol .call_test_bool>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $x3, implicit $x2, implicit-def $r1, implicit-def $x3
; 64BIT:          ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

@f1 = global float 0.000000e+00, align 4
@d1 = global double 0.000000e+00, align 8

define void @call_test_floats() {
entry:
  %0 = load float, float* @f1, align 4
  call float @test_floats(float %0, float %0, float %0)
  ret void
}

; CHECK-LABEL: name: call_test_floats{{.*}}

; 32BIT:      renamable $r3 = LWZtoc @f1, $r2 :: (load 4 from got)
; 32BIT-NEXT: renamable $f1 = LFS 0, killed renamable $r3 :: (dereferenceable load 4 from @f1)
; 32BIT-NEXT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT: $f2 = COPY renamable $f1
; 32BIT-NEXT: $f3 = COPY renamable $f1
; 32BIT-NEXT: BL_NOP <mcsymbol .test_floats>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $f1, implicit killed $f2, implicit killed $f3, implicit $r2, implicit-def $r1
; 32BIT-NEXT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT:      renamable $x3 = LDtoc @f1, $x2 :: (load 8 from got)
; 64BIT-NEXT: renamable $f1 = LFS 0, killed renamable $x3 :: (dereferenceable load 4 from @f1)
; 64BIT-NEXT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT: $f2 = COPY renamable $f1
; 64BIT-NEXT: $f3 = COPY renamable $f1
; 64BIT-NEXT: BL8_NOP <mcsymbol .test_floats>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $f1, implicit killed $f2, implicit killed $f3, implicit $x2, implicit-def $r1
; 64BIT-NEXT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

define float @test_floats(float %f1, float %f2, float %f3) {
entry:
  %add = fadd float %f1, %f2
  %add1 = fadd float %add, %f3
  ret float %add1
}

; CHECK-LABEL: name: test_floats{{.*}}

; CHECK:      liveins:
; CHECK-NEXT: - { reg: '$f1', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f2', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f3', virtual-reg: '' }
; CHECK:      body:             |
; CHECK-NEXT:   bb.0.entry:
; CHECK-NEXT:     liveins: $f1, $f2, $f3

define void @call_test_fpr_max() {
entry:
  %0 = load double, double* @d1, align 8
  call double @test_fpr_max(double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0, double %0)
  ret void
}

; CHECK-LABEL: name: call_test_fpr_max{{.*}}

; 32BIT:      renamable $r3 = LWZtoc @d1, $r2 :: (load 4 from got)
; 32BIT-NEXT: renamable $f1 = LFD 0, killed renamable $r3 :: (dereferenceable load 8 from @d1)
; 32BIT-NEXT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT: $f2 = COPY renamable $f1
; 32BIT-NEXT: $f3 = COPY renamable $f1
; 32BIT-NEXT: $f4 = COPY renamable $f1
; 32BIT-NEXT: $f5 = COPY renamable $f1
; 32BIT-NEXT: $f6 = COPY renamable $f1
; 32BIT-NEXT: $f7 = COPY renamable $f1
; 32BIT-NEXT: $f8 = COPY renamable $f1
; 32BIT-NEXT: $f9 = COPY renamable $f1
; 32BIT-NEXT: $f10 = COPY renamable $f1
; 32BIT-NEXT: $f11 = COPY renamable $f1
; 32BIT-NEXT: $f12 = COPY renamable $f1
; 32BIT-NEXT: $f13 = COPY renamable $f1
; 32BIT-NEXT: BL_NOP <mcsymbol .test_fpr_max>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $f1, implicit killed $f2, implicit killed $f3, implicit killed $f4, implicit killed $f5, implicit killed $f6, implicit killed $f7, implicit killed $f8, implicit killed $f9, implicit killed $f10, implicit killed $f11, implicit killed $f12, implicit killed $f13, implicit $r2, implicit-def $r1
; 32BIT-NEXT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT:      renamable $x3 = LDtoc @d1, $x2 :: (load 8 from got)
; 64BIT-NEXT: renamable $f1 = LFD 0, killed renamable $x3 :: (dereferenceable load 8 from @d1)
; 64BIT-NEXT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT: $f2 = COPY renamable $f1
; 64BIT-NEXT: $f3 = COPY renamable $f1
; 64BIT-NEXT: $f4 = COPY renamable $f1
; 64BIT-NEXT: $f5 = COPY renamable $f1
; 64BIT-NEXT: $f6 = COPY renamable $f1
; 64BIT-NEXT: $f7 = COPY renamable $f1
; 64BIT-NEXT: $f8 = COPY renamable $f1
; 64BIT-NEXT: $f9 = COPY renamable $f1
; 64BIT-NEXT: $f10 = COPY renamable $f1
; 64BIT-NEXT: $f11 = COPY renamable $f1
; 64BIT-NEXT: $f12 = COPY renamable $f1
; 64BIT-NEXT: $f13 = COPY renamable $f1
; 64BIT-NEXT: BL8_NOP <mcsymbol .test_fpr_max>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $f1, implicit killed $f2, implicit killed $f3, implicit killed $f4, implicit killed $f5, implicit killed $f6, implicit killed $f7, implicit killed $f8, implicit killed $f9, implicit killed $f10, implicit killed $f11, implicit killed $f12, implicit killed $f13, implicit $x2, implicit-def $r1
; 64BIT-NEXT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

define double @test_fpr_max(double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, double %d8, double %d9, double %d10, double %d11, double %d12, double %d13) {
entry:
  %add = fadd double %d1, %d2
  %add1 = fadd double %add, %d3
  %add2 = fadd double %add1, %d4
  %add3 = fadd double %add2, %d5
  %add4 = fadd double %add3, %d6
  %add5 = fadd double %add4, %d7
  %add6 = fadd double %add5, %d8
  %add7 = fadd double %add6, %d9
  %add8 = fadd double %add7, %d10
  %add9 = fadd double %add8, %d11
  %add10 = fadd double %add9, %d12
  %add11 = fadd double %add10, %d13
  ret double %add11
}

; CHECK-LABEL: name: test_fpr_max{{.*}}

; CHECK:      liveins:
; CHECK-NEXT: - { reg: '$f1', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f2', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f3', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f4', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f5', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f6', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f7', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f8', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f9', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f10', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f11', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f12', virtual-reg: '' }
; CHECK-NEXT: - { reg: '$f13', virtual-reg: '' }
; CHECK:      body:             |
; CHECK-NEXT:   bb.0.entry:
; CHECK-NEXT:     liveins: $f1, $f2, $f3, $f4, $f5, $f6, $f7, $f8, $f9, $f10, $f11, $f12, $f13

define void @call_test_mix() {
entry:
  %0 = load float, float* @f1, align 4
  %1 = load double, double* @d1, align 8
  call i32 @test_mix(float %0, i32 1, double %1, i8 signext 97)
  ret void
}

; CHECK-LABEL: name: call_test_mix{{.*}}

; 32BIT:      renamable $r[[REG1:[0-9]+]] = LWZtoc @f1, $r2 :: (load 4 from got)
; 32BIT-NEXT: renamable $r[[REG2:[0-9]+]] = LWZtoc @d1, $r2 :: (load 4 from got)
; 32BIT-NEXT: renamable $f1 = LFS 0, killed renamable $r[[REG1]] :: (dereferenceable load 4 from @f1)
; 32BIT-NEXT: renamable $f2 = LFD 0, killed renamable $r[[REG2]] :: (dereferenceable load 8 from @d1)
; 32BIT-NEXT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT-NEXT: $r4 = LI 1
; 32BIT-NEXT: $r7 = LI 97
; 32BIT-NEXT: BL_NOP <mcsymbol .test_mix>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $f1, implicit $r4, implicit $f2, implicit killed $r7, implicit $r2, implicit-def $r1
; 32BIT-NEXT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT:      renamable $x[[REG1:[0-9]+]] = LDtoc @f1, $x2 :: (load 8 from got)
; 64BIT-NEXT: renamable $x[[REG2:[0-9]+]] = LDtoc @d1, $x2 :: (load 8 from got)
; 64BIT-NEXT: renamable $f1 = LFS 0, killed renamable $x[[REG1]] :: (dereferenceable load 4 from @f1)
; 64BIT-NEXT: renamable $f2 = LFD 0, killed renamable $x[[REG2]] :: (dereferenceable load 8 from @d1)
; 64BIT-NEXT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT-NEXT: $x4 = LI8 1
; 64BIT-NEXT: $x6 = LI8 97
; 64BIT-NEXT: BL8_NOP <mcsymbol .test_mix>, csr_aix64, implicit-def dead $lr8, implicit $rm, implicit $f1, implicit $x4, implicit $f2, implicit killed $x6, implicit $x2, implicit-def $r1
; 64BIT-NEXT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

define i32 @test_mix(float %f, i32 signext %i, double %d, i8 signext %c) {
entry:
  %conv = fpext float %f to double
  %add = fadd double %conv, %d
  %conv1 = fptrunc double %add to float
  %conv2 = zext i8 %c to i32
  %add3 = add nsw i32 %i, %conv2
  %conv4 = sitofp i32 %add3 to float
  %add5 = fadd float %conv4, %conv1
  %conv6 = fptosi float %add5 to i32
  ret i32 %conv6
}

; CHECK-LABEL: name: test_mix{{.*}}

; 32BIT:      liveins:
; 32BIT-NEXT: - { reg: '$f1', virtual-reg: '' }
; 32BIT-NEXT: - { reg: '$r4', virtual-reg: '' }
; 32BIT-NEXT: - { reg: '$f2', virtual-reg: '' }
; 32BIT-NEXT: - { reg: '$r7', virtual-reg: '' }
; 32BIT:      body:             |
; 32BIT-NEXT:   bb.0.entry:
; 32BIT-NEXT:     liveins: $f1, $f2, $r4, $r7

; 64BIT:      liveins:
; 64BIT-NEXT: - { reg: '$f1', virtual-reg: '' }
; 64BIT-NEXT: - { reg: '$x4', virtual-reg: '' }
; 64BIT-NEXT: - { reg: '$f2', virtual-reg: '' }
; 64BIT-NEXT: - { reg: '$x6', virtual-reg: '' }
; 64BIT:      body:             |
; 64BIT-NEXT:   bb.0.entry:
; 64BIT-NEXT:     liveins: $f1, $f2, $x4, $x6


define i64 @callee_mixed_ints(i32 %a, i8 signext %b, i32 %c, i16 signext %d, i64 %e) {
entry:
  %conv = zext i8 %b to i32
  %add = add nsw i32 %a, %conv
  %add1 = add nsw i32 %add, %c
  %conv2 = sext i16 %d to i32
  %add3 = add nsw i32 %add1, %conv2
  %conv4 = sext i32 %add3 to i64
  %add5 = add nsw i64 %conv4, %e
  ret i64 %add5
  }

; CHECK-LABEL: name:  callee_mixed_ints

; 32BIT:      liveins:
; 32BIT-NEXT: - { reg: '$r3', virtual-reg: '' }
; 32BIT-NEXT: - { reg: '$r4', virtual-reg: '' }
; 32BIT-NEXT: - { reg: '$r5', virtual-reg: '' }
; 32BIT-NEXT: - { reg: '$r6', virtual-reg: '' }
; 32BIT-NEXT: - { reg: '$r7', virtual-reg: '' }
; 32BIT-NEXT: - { reg: '$r8', virtual-reg: '' }
; 32BIT:      body:             |
; 32BIT-NEXT:  bb.0.entry:
; 32BIT-NEXT:   liveins: $r3, $r4, $r5, $r6, $r7, $r8
  
; 64BIT:        liveins:
; 64BIT-NEXT:   - { reg: '$x3', virtual-reg: '' }
; 64BIT-NEXT:   - { reg: '$x4', virtual-reg: '' }
; 64BIT-NEXT:   - { reg: '$x5', virtual-reg: '' }
; 64BIT-NEXT:   - { reg: '$x6', virtual-reg: '' }
; 64BIT-NEXT:   - { reg: '$x7', virtual-reg: '' }
; 64BIT:        body:             |
; 64BIT-NEXT:    bb.0.entry:
; 64BIT-NEXT:     liveins: $x3, $x4, $x5, $x6, $x7
