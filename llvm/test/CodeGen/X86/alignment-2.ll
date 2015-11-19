; RUN: llc < %s -mtriple i386-apple-darwin10 | FileCheck %s
; <rdar://problem/10058036>

%struct._psqlSettings = type { %struct.pg_conn*, i32, %struct.__sFILE*, i8, %struct.printQueryOpt, i8*, i8, i32, %struct.__sFILE*, i8, i32, i8*, i8*, i8*, i64, i8, %struct.__sFILE*, %struct._variable*, i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, i8*, i8*, i8*, i32 }
%struct.pg_conn = type opaque
%struct.__sFILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sbuf = type { i8*, i32 }
%struct.__sFILEX = type opaque
%struct.printQueryOpt = type { %struct.printTableOpt, i8*, i8, i8*, i8**, i8, i8, i8* }
%struct.printTableOpt = type { i32, i8, i16, i16, i8, i8, i8, i32, %struct.printTextFormat*, i8*, i8*, i8, i8*, i32, i32, i32 }
%struct.printTextFormat = type { i8*, [4 x %struct.printTextLineFormat], i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8 }
%struct.printTextLineFormat = type { i8*, i8*, i8*, i8* }
%struct._variable = type { i8*, i8*, void (i8*)*, %struct._variable* }
%struct.pg_result = type opaque

@pset = external global %struct._psqlSettings

define signext i8 @do_lo_list() nounwind optsize ssp {
bb:
; CHECK:     do_lo_list
; Make sure we do not use movaps for the global variable.
; It is okay to use movaps for writing the local variable on stack.
; CHECK-NOT: movaps {{[0-9]*}}(%{{[a-z]*}}), {{%xmm[0-9]}}
  %myopt = alloca %struct.printQueryOpt, align 4
  %tmp = bitcast %struct.printQueryOpt* %myopt to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp, i8* bitcast (%struct.printQueryOpt* getelementptr inbounds (%struct._psqlSettings, %struct._psqlSettings* @pset, i32 0, i32 4) to i8*), i32 76, i32 4, i1 false)
  ret i8 0
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
