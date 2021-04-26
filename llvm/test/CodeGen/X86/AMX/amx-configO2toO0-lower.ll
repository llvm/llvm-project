; RUN: opt < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-int8 -mattr=+avx512f -lower-amx-type -S | FileCheck %s

@buf = dso_local global [1024 x i8] zeroinitializer, align 16
@buf2 = dso_local global [1024 x i8] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define dso_local void @test_api(i32 %cond, i16 signext %row, i16 signext %col) local_unnamed_addr {

; CHECK-LABEL: entry:
; CHECK:        %{{[0-9]+}} = alloca <256 x i32>, align 1024
; CHECK-NEXT:   %{{[0-9]+}} = bitcast <256 x i32>* %{{[0-9]+}} to i8*
; CHECK-NEXT:   %{{[0-9]+}} = alloca <256 x i32>, align 1024
; CHECK-NEXT:   %{{[0-9]+}} = bitcast <256 x i32>* %{{[0-9]+}} to i8*
; CHECK-NEXT:   %{{[0-9]+}} = alloca <256 x i32>, align 1024
; CHECK-NEXT:   %{{[0-9]+}} = bitcast <256 x i32>* %{{[0-9]+}} to i8*
; CHECK-NEXT:   %{{[0-9]+}} = alloca <256 x i32>, align 1024
; CHECK-NEXT:   %{{[0-9]+}} = bitcast <256 x i32>* %{{[0-9]+}} to i8*
; CHECK-NEXT:   %tobool.not = icmp eq i32 %cond, 0
; CHECK-NEXT:   br i1 %tobool.not, label %if.else, label %if.then
; CHECK:      if.then:
; CHECK-NEXT:   %{{[0-9]+}} = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 8, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf, i64 0, i64 0), i64 32)
; CHECK-NEXT:   call void @llvm.x86.tilestored64.internal(i16 %row, i16 8, i8* %{{[0-9]+}}, i64 64, x86_amx %{{[0-9]+}})
; CHECK-NEXT:   %{{[0-9]+}} = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 8, i16 %col, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf, i64 0, i64 0), i64 32)
; CHECK-NEXT:   call void @llvm.x86.tilestored64.internal(i16 8, i16 %col, i8* %{{[0-9]+}}, i64 64, x86_amx %{{[0-9]+}})
; CHECK-NEXT:   %{{[0-9]+}} = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 %col, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf, i64 0, i64 0), i64 32)
; CHECK-NEXT:   call void @llvm.x86.tilestored64.internal(i16 %row, i16 %col, i8* %{{[0-9]+}}, i64 64, x86_amx %{{[0-9]+}})
; CHECK-NEXT:   br label %if.end
; CHECK:      if.else:
; CHECK-NEXT:   %{{[0-9]+}} = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 8, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf2, i64 0, i64 0), i64 32)
; CHECK-NEXT:   call void @llvm.x86.tilestored64.internal(i16 %row, i16 8, i8* %{{[0-9]+}}, i64 64, x86_amx %{{[0-9]+}})
; CHECK-NEXT:   %{{[0-9]+}} = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 8, i16 %col, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf2, i64 0, i64 0), i64 32)
; CHECK-NEXT:   call void @llvm.x86.tilestored64.internal(i16 8, i16 %col, i8* %{{[0-9]+}}, i64 64, x86_amx %{{[0-9]+}})
; CHECK-NEXT:   %{{[0-9]+}} = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 %col, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf2, i64 0, i64 0), i64 32)
; CHECK-NEXT:   call void @llvm.x86.tilestored64.internal(i16 %row, i16 %col, i8* %{{[0-9]+}}, i64 64, x86_amx %{{[0-9]+}})
; CHECK-NEXT:   br label %if.end
; CHECK:      if.end:
; CHECK-NEXT:   %{{[0-9]+}} = call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 8, i8* %{{[0-9]+}}, i64 64)
; CHECK-NEXT:   %{{[0-9]+}} = call x86_amx @llvm.x86.tileloadd64.internal(i16 8, i16 %col, i8* %{{[0-9]+}}, i64 64)
; CHECK-NEXT:   %{{[0-9]+}} = call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 %col, i8* %{{[0-9]+}}, i64 64)
; CHECK-NEXT:   %{{[0-9]+}} = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 %row, i16 %col, i16 8, x86_amx %{{[0-9]+}}, x86_amx %{{[0-9]+}}, x86_amx %{{[0-9]+}})
; CHECK-NEXT:   call void @llvm.x86.tilestored64.internal(i16 %row, i16 %col, i8* %{{[0-9]+}}, i64 64, x86_amx %{{[0-9]+}})
; CHECK-NEXT:   %{{[0-9]+}} = call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 %col, i8* %{{[0-9]+}}, i64 64)
; CHECK-NEXT:   tail call void @llvm.x86.tilestored64.internal(i16 %row, i16 %col, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf, i64 0, i64 0), i64 32, x86_amx %{{[0-9]+}})
; CHECK-NEXT:   ret void

entry:
  %tobool.not = icmp eq i32 %cond, 0
  br i1 %tobool.not, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %0 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 8, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf, i64 0, i64 0), i64 32)
  %1 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 8, i16 %col, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf, i64 0, i64 0), i64 32)
  %2 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 %col, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf, i64 0, i64 0), i64 32)
  br label %if.end

if.else:                                          ; preds = %entry
  %3 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 8, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf2, i64 0, i64 0), i64 32)
  %4 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 8, i16 %col, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf2, i64 0, i64 0), i64 32)
  %5 = tail call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 %col, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf2, i64 0, i64 0), i64 32)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %a.sroa.1094.0.in = phi x86_amx [ %3, %if.else ], [ %0, %if.then ]
  %b.sroa.1069.0.in = phi x86_amx [ %4, %if.else ], [ %1, %if.then ]
  %c.sroa.1044.0.in = phi x86_amx [ %5, %if.else ], [ %2, %if.then ]
  %6 = tail call x86_amx @llvm.x86.tdpbssd.internal(i16 %row, i16 %col, i16 8, x86_amx %c.sroa.1044.0.in, x86_amx %a.sroa.1094.0.in, x86_amx %b.sroa.1069.0.in)
  tail call void @llvm.x86.tilestored64.internal(i16 %row, i16 %col, i8* getelementptr inbounds ([1024 x i8], [1024 x i8]* @buf, i64 0, i64 0), i64 32, x86_amx %6)
  ret void
}

; Function Attrs: nounwind
declare x86_amx @llvm.x86.tileloadd64.internal(i16, i16, i8*, i64)

; Function Attrs: nounwind
declare x86_amx @llvm.x86.tdpbssd.internal(i16, i16, i16, x86_amx, x86_amx, x86_amx)

; Function Attrs: nounwind
declare void @llvm.x86.tilestored64.internal(i16, i16, i8*, i64, x86_amx)
