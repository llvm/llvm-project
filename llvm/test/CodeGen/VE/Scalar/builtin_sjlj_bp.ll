; RUN: llc < %s -mtriple=ve | FileCheck %s

%Foo = type { [125 x i8] }
declare void @whatever(i64, %Foo*, i8**, i8*, i8*, i32)  #0
declare i32 @llvm.eh.sjlj.setjmp(i8*) nounwind

; Function Attrs: noinline nounwind optnone
define i32 @t_setjmp(i64 %n, %Foo* byval(%Foo) nocapture readnone align 8 %f) {
; CHECK-LABEL: t_setjmp:
; CHECK:       .LBB{{[0-9]+}}_5:
; CHECK-NEXT:    st %s18, 48(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s19, 56(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s20, 64(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s21, 72(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s22, 80(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s23, 88(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s24, 96(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s25, 104(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s26, 112(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s27, 120(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s28, 128(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s29, 136(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s30, 144(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s31, 152(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s32, 160(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s33, 168(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    st %s1, 312(, %s17) # 8-byte Folded Spill
; CHECK-NEXT:    st %s0, 304(, %s17) # 8-byte Folded Spill
; CHECK-NEXT:    lea %s0, 15(, %s0)
; CHECK-NEXT:    and %s0, -16, %s0
; CHECK-NEXT:    lea %s1, __ve_grow_stack@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __ve_grow_stack@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    lea %s1, 240(, %s11)
; CHECK-NEXT:    st %s1, 328(, %s17)
; CHECK-NEXT:    lea %s0, .LBB{{[0-9]+}}_3@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, .LBB{{[0-9]+}}_3@hi(, %s0)
; CHECK-NEXT:    st %s17, 24(, %s1)
; CHECK-NEXT:    st %s1, 296(, %s17) # 8-byte Folded Spill
; CHECK-NEXT:    st %s0, 8(, %s1)
; CHECK-NEXT:    # EH_SJlJ_SETUP .LBB{{[0-9]+}}_3
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    lea %s5, 0
; CHECK-NEXT:    br.l.t .LBB{{[0-9]+}}_2
; CHECK-NEXT:  .LBB{{[0-9]+}}_3: # Block address taken
; CHECK-NEXT:    ld %s17, 24(, %s10)
; CHECK-NEXT:    lea %s5, 1
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, whatever@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, whatever@hi(, %s0)
; CHECK-NEXT:    lea %s2, 328(, %s17)
; CHECK-NEXT:    lea %s3, 320(, %s17)
; CHECK-NEXT:    ld %s0, 304(, %s17) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s1, 312(, %s17) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s4, 296(, %s17) # 8-byte Folded Reload
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    ld %s33, 168(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s32, 160(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s31, 152(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s30, 144(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s29, 136(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s28, 128(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s27, 120(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s26, 112(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s25, 104(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s24, 96(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s23, 88(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s22, 80(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s21, 72(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s20, 64(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s19, 56(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s18, 48(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    or %s11, 0, %s9
  %buf = alloca [5 x i8*], align 16
  %p = alloca i8*, align 8
  %q = alloca i8, align 64
  %r = bitcast [5 x i8*]* %buf to i8*
  %s = alloca i8, i64 %n, align 1
  store i8* %s, i8** %p, align 8
  %t = call i32 @llvm.eh.sjlj.setjmp(i8* %s)
  call void @whatever(i64 %n, %Foo* %f, i8** %p, i8* %q, i8* %s, i32 %t) #1
  ret i32 0
}
