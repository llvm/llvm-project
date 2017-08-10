; RUN: llc < %s -mtriple=i686-- -mcpu=atom | FileCheck %s
; CHECK:        movl
; CHECK:        movb
; CHECK:        movb
; CHECK:        cmpb
; CHECK:        notb
; CHECK:        notb

; Test for checking of cancel conversion to cmp32 in Atom case 
; in function 'X86TargetLowering::EmitCmp'
 
define i8 @run_test(i8* %rd_p) {
entry:
  %incdec.ptr = getelementptr inbounds i8, i8* %rd_p, i64 1
  %ld1 = load i8, i8* %rd_p, align 1
  %incdec.ptr1 = getelementptr inbounds i8, i8* %rd_p, i64 2
  %ld2 = load i8, i8* %incdec.ptr, align 1
  %x4 = xor i8 %ld1, -1
  %x5 = xor i8 %ld2, -1
  %cmp34 = icmp ult i8 %ld2, %ld1
  br i1 %cmp34, label %if.then3, label %if.else
 
if.then3:                                        
  %sub7 = sub i8 %x4, %x5
  br label %if.end4
 
if.else:                           
  %sub8 = sub i8 %x5, %x4 
  br label %if.end4 
 
if.end4:          
  %res = phi i8 [ %sub7, %if.then3 ], [ %sub8, %if.else ]
  ret i8 %res

}

