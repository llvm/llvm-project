; Test the passing of scalar values in GPRs, FPRs in 64-bit calls on z/OS.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z10 | FileCheck %s

define i8 @call_char(){
; CHECK-LABEL: call_char DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,8(5)
; CHECK-NEXT:    lg 5,0(5)
; CHECK-NEXT:    lghi 1,8
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
  %retval = call i8 (i8) @pass_char(i8 8)
  ret i8 %retval
}

define i16 @call_short() {
; CHECK-LABEL: call_short DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,24(5)
; CHECK-NEXT:    lg 5,16(5)
; CHECK-NEXT:    lghi 1,16
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
entry:
  %retval = call i16 (i16) @pass_short(i16 16)
  ret i16 %retval
}

define i32 @call_int() {
; CHECK-LABEL: call_int DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,40(5)
; CHECK-NEXT:    lg 5,32(5)
; CHECK-NEXT:    lghi 1,32
; CHECK-NEXT:    lghi 2,33
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
entry:
  %retval = call i32 (i32, i32) @pass_int(i32 32, i32 33)
  ret i32 %retval
}

define i64 @call_long() {
; CHECK-LABEL: call_long DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,56(5)
; CHECK-NEXT:    lg 5,48(5)
; CHECK-NEXT:    lghi 1,64
; CHECK-NEXT:    lghi 2,65
; CHECK-NEXT:    lghi 3,66
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
entry:
  %retval = call i64 (i64, i64, i64) @pass_long(i64 64, i64 65, i64 66)
  ret i64 %retval
}

define i32 @call_ptr(ptr %p1, ptr %p2) {
; CHECK-LABEL: call_ptr DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,72(5)
; CHECK-NEXT:    lg 5,64(5)
; CHECK-NEXT:    lgr 1,2
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
entry:
  %retval = call i32 (ptr) @pass_ptr(ptr %p2)
  ret i32 %retval
}

define i64 @call_integrals() {
; CHECK-LABEL: call_integrals DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,88(5)
; CHECK-NEXT:    lg 5,80(5)
; CHECK-NEXT:    lghi 1,64
; CHECK-NEXT:    lghi 2,32
; CHECK-NEXT:    lghi 3,16
; CHECK-NEXT:    mvghi 2200(4),128
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
entry:
  %retval = call i64 (i64, i32, i16, i64) @pass_integrals0(i64 64, i32 32, i16 16, i64 128)
  ret i64 %retval
}

define signext i8 @pass_char(i8 signext %arg) {
; CHECK-LABEL: pass_char DS 0H
; CHECK:         lgr 3,1
; CHECK-NEXT:    b 2(7)
entry:
  ret i8 %arg
}

define signext i16 @pass_short(i16 signext %arg) {
; CHECK-LABEL: pass_short DS 0H
; CHECK:         lgr 3,1
; CHECK-NEXT:    b 2(7)
entry:
  ret i16 %arg
}

define signext i32 @pass_int(i32 signext %arg0, i32 signext %arg1) {
; CHECK-LABEL: pass_int DS 0H
; CHECK:         lgr 3,2
; CHECK-NEXT:    b 2(7)
entry:
  ret i32 %arg1
}

define signext i64 @pass_long(i64 signext %arg0, i64 signext %arg1, i64 signext %arg2) {
; CHECK-LABEL: pass_long DS 0H
; CHECK:         agr 1,2
; CHECK-NEXT:    agr 3,1
; CHECK-NEXT:    b 2(7)
entry:
  %N = add i64 %arg0, %arg1
  %M = add i64 %N, %arg2
  ret i64 %M
}

define signext i64 @pass_integrals0(i64 signext %arg0, i32 signext %arg1, i16 signext %arg2, i64 signext %arg3) {
; CHECK-LABEL: pass_integrals0 DS 0H
; CHECK:         ag 2,2200(4)
; CHECK-NEXT:    lgr 3,2
; CHECK-NEXT:    b 2(7)
entry:
  %N = sext i32 %arg1 to i64
  %M = add i64 %arg3, %N
  ret i64 %M
}

define float @call_float() {
; CHECK-LABEL: call_float DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,104(5)
; CHECK-NEXT:    lg 5,96(5)
; CHECK-NEXT:    larl 1,L#CPI11_0
; CHECK-NEXT:    le 0,0(1)
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
entry:
  %ret = call float (float) @pass_float(float 0x400921FB60000000)
  ret float %ret
}

define double @call_double() {
; CHECK-LABEL: call_double DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,120(5)
; CHECK-NEXT:    lg 5,112(5)
; CHECK-NEXT:    larl 1,L#{{CPI[0-9]+_[0-9]+}}
; CHECK-NEXT:    ld 0,0(1)
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
entry:
  %ret = call double (double) @pass_double(double 3.141000e+00)
  ret double %ret
}

define fp128 @call_longdouble() {
; CHECK-LABEL: call_longdouble DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,136(5)
; CHECK-NEXT:    lg 5,128(5)
; CHECK-NEXT:    larl 1,L#{{CPI[0-9]+_[0-9]+}}
; CHECK-NEXT:    ld 0,0(1)
; CHECK-NEXT:    ld 2,8(1)
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
entry:
  %ret = call fp128 (fp128) @pass_longdouble(fp128 0xLE0FC1518450562CD4000921FB5444261)
  ret fp128 %ret
}

define i64 @call_floats0(fp128 %arg0, double %arg1) {
; CHECK-LABEL: call_floats0 DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,152(5)
; CHECK-NEXT:    lg 5,144(5)
; CHECK-NEXT:    larl 1,L#{{CPI[0-9]+_[0-9]+}}
; CHECK-NEXT:    ld  1,0(1)
; CHECK-NEXT:    ld  3,8(1)
; CHECK-NEXT:    lxr 5,0
; CHECK-NEXT:    std 4,2208(4)
; CHECK-NEXT:    lxr 0,1
; CHECK-NEXT:    lxr 4,5
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
entry:
  %ret = call i64 (fp128, fp128, double) @pass_floats0(fp128 0xLE0FC1518450562CD4000921FB5444261, fp128 %arg0, double %arg1)
  ret i64 %ret
}

define i64 @call_floats1(fp128 %arg0, double %arg1) {
; CHECK-LABEL: call_floats1 DS 0H
; CHECK:         stmg 6,7,1872(4)
; CHECK-NEXT:  L#stack_update{{[0-9]+}} DS 0H                                                           
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    *FENCE
; CHECK-NEXT:  L#end_of_prologue{{[0-9]+}} DS 0H                                                        
; CHECK-NEXT:    lg 6,168(5)
; CHECK-NEXT:    lg 5,160(5)
; CHECK-NEXT:    lxr 1,0
; CHECK-NEXT:    ldr 0,4
; CHECK-NEXT:    lxr 4,1
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    lg 7,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
entry:
  %ret = call i64 (double, fp128) @pass_floats1(double %arg1, fp128 %arg0)
  ret i64 %ret
}

define float @pass_float(float %arg) {
; CHECK-LABEL: pass_float DS 0H
; CHECK:         larl  1,L#{{CPI[0-9]+_[0-9]+}}
; CHECK-NEXT:    aeb 0,0(1)
; CHECK-NEXT:    b 2(7)
entry:
  %X = fadd float %arg, 0x400821FB60000000
  ret float %X
}

define double @pass_double(double %arg) {
; CHECK-LABEL: pass_double DS 0H
; CHECK:         larl  1,L#{{CPI[0-9]+_[0-9]+}}
; CHECK-NEXT:    adb 0,0(1)
; CHECK-NEXT:    b 2(7)
entry:
  %X = fadd double %arg, 1.414213e+00
  ret double %X
}

define fp128 @pass_longdouble(fp128 %arg) {
; CHECK-LABEL: pass_longdouble DS 0H
; CHECK:         larl  1,L#{{CPI[0-9]+_[0-9]+}}
; CHECK-NEXT:    lxdb 1,0(1)
; CHECK-NEXT:    axbr 0,1
; CHECK-NEXT:    b 2(7)
entry:
  %X = fadd fp128 %arg, 0xL10000000000000004000921FB53C8D4F
  ret fp128 %X
}

define i64 @pass_floats0(fp128 %arg0, fp128 %arg1, double %arg2) {
; CHECK-LABEL: pass_floats0 DS 0H
; CHECK:         lxdb 1,2208(4)
; CHECK:         larl  1,L#{{CPI[0-9]+_[0-9]+}}
; CHECK-NEXT:    ld 5,0(1)
; CHECK-NEXT:    ld 7,8(1)
; CHECK-NEXT:    axbr 0,4
; CHECK-NEXT:    axbr 1,0
; CHECK-NEXT:    cxbr 1,5
; CHECK-NEXT:    ipm 0
; CHECK-NEXT:    afi 0,-268435456
; CHECK-NEXT:    sllg 0,0,34
; CHECK-NEXT:    srag 3,0,63
; CHECK-NEXT:    b 2(7)
  %X = fadd fp128 %arg0, %arg1
  %arg2_ext = fpext double %arg2 to fp128
  %Y = fadd fp128 %X, %arg2_ext
  %ret_bool = fcmp ueq fp128 %Y, 0xLE0FC1518450562CD4000921FB5444261
  %ret = sext i1 %ret_bool to i64
  ret i64 %ret
}

declare i64 @pass_floats1(double %arg0, fp128 %arg1)
declare i32 @pass_ptr(ptr %arg)
