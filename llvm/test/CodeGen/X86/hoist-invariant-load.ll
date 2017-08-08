; REQUIRES: asserts
; RUN: llc -mcpu=haswell < %s -stats -O2 2>&1 | grep "4 machinelicm.*hoisted"
; For test:
; 2 invariant loads, 1 for OBJC_SELECTOR_REFERENCES_
; and 1 for objc_msgSend from the GOT
; For test_multi_def:
; 2 invariant load (full multiply, both loads should be hoisted.)

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.2"

@"\01L_OBJC_METH_VAR_NAME_" = internal global [4 x i8] c"foo\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal global i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i64 0, i64 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_IMAGE_INFO" = internal constant [2 x i32] [i32 0, i32 16], section "__DATA, __objc_imageinfo, regular, no_dead_strip"
@llvm.used = appending global [3 x i8*] [i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_" to i8*), i8* bitcast ([2 x i32]* @"\01L_OBJC_IMAGE_INFO" to i8*)], section "llvm.metadata"

define void @test(i8* %x) uwtable ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8, !invariant.load !0
  %call = tail call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %x, i8* %0)
  %inc = add i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare i8* @objc_msgSend(i8*, i8*, ...) nonlazybind

define void @test_multi_def(i64* dereferenceable(8) %x1,
                            i64* dereferenceable(8) %x2,
                            i128* %y, i64 %count) nounwind {
entry:
  br label %for.body

for.check:
  %inc = add nsw i64 %i, 1
  %done = icmp sge i64 %inc, %count
  br i1 %done, label %exit, label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %inc, %for.check ]
  %x1_load = load i64, i64* %x1, align 8, !invariant.load !0
  %x1_zext = zext i64 %x1_load to i128
  %x2_load = load i64, i64* %x2, align 8, !invariant.load !0
  %x2_zext = zext i64 %x2_load to i128
  %x_prod = mul i128 %x1_zext, %x2_zext
  %y_elem = getelementptr inbounds i128, i128* %y, i64 %i
  %y_load = load i128, i128* %y_elem, align 8
  %y_plus = add i128 %x_prod, %y_load
  store i128 %y_plus, i128* %y_elem, align 8
  br label %for.check

exit:
  ret void
}

!0 = !{}
