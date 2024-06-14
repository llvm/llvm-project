; RUN: llc < %s -mtriple i386-apple-darwin10 | FileCheck %s
; <rdar://problem/10058036>

%struct._psqlSettings = type { ptr, i32, ptr, i8, %struct.printQueryOpt, ptr, i8, i32, ptr, i8, i32, ptr, ptr, ptr, i64, i8, ptr, ptr, i8, i8, i8, i8, i8, i32, i32, i32, i32, i32, ptr, ptr, ptr, i32 }
%struct.pg_conn = type opaque
%struct.__sFILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sbuf = type { ptr, i32 }
%struct.__sFILEX = type opaque
%struct.printQueryOpt = type { %struct.printTableOpt, ptr, i8, ptr, ptr, i8, i8, ptr }
%struct.printTableOpt = type { i32, i8, i16, i16, i8, i8, i8, i32, ptr, ptr, ptr, i8, ptr, i32, i32, i32 }
%struct.printTextFormat = type { ptr, [4 x %struct.printTextLineFormat], ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8 }
%struct.printTextLineFormat = type { ptr, ptr, ptr, ptr }
%struct._variable = type { ptr, ptr, ptr, ptr }
%struct.pg_result = type opaque

@pset = external global %struct._psqlSettings

define signext i8 @do_lo_list() nounwind optsize ssp {
bb:
; CHECK:     do_lo_list
; Make sure we do not use movaps for the global variable.
; It is okay to use movaps for writing the local variable on stack.
; CHECK-NOT: movaps {{[0-9]*}}(%{{[a-z]*}}), {{%xmm[0-9]}}
  %myopt = alloca %struct.printQueryOpt, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %myopt, ptr align 4 getelementptr inbounds (%struct._psqlSettings, ptr @pset, i32 0, i32 4), i32 76, i1 false)
  ret i8 0
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
