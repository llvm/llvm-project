; RUN: not llc -mtriple=bpf -mcpu=v3 < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: error: <unknown>:0:0: in function foo i64 (i32, i32, i32): {{(0x[0-9a-fA-F]+|t[0-9]+)}}: i64 = GlobalAddress<ptr @bar> 0 pass by value not supported

; Source code:
;   struct t { long a; long b; long c;};
;
;   long bar(int a1, int a2, int a3, int a4, int a5, struct t a6);
;   long foo(int a1, int a2, int a3) {
;     struct t tmp = {a1, a2, a3};
;     return bar(a1, a2, a3, a2, a1, tmp);
;   }

%struct.t = type { i64, i64, i64 }

define dso_local i64 @foo(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr {
  %4 = alloca %struct.t, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  %5 = sext i32 %0 to i64
  store i64 %5, ptr %4, align 8
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %7 = sext i32 %1 to i64
  store i64 %7, ptr %6, align 8
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %9 = sext i32 %2 to i64
  store i64 %9, ptr %8, align 8
  %10 = tail call i64 @bar(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %1, i32 noundef %0, ptr noundef nonnull byval(%struct.t) align 8 %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  ret i64 %10
}

declare void @llvm.lifetime.start.p0(ptr captures(none))

declare dso_local i64 @bar(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, ptr noundef byval(%struct.t) align 8) local_unnamed_addr

declare void @llvm.lifetime.end.p0(ptr captures(none))
