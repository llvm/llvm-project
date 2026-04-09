; Check the volatile load/stores are not widened by HexagonLoadStoreWidening pass

; RUN: llc -mtriple=hexagon -verify-machineinstrs < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: volatile_loads:
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}+#0)
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}+#4)
; CHECK-NOT: r{{[0-9]+}} = memd(r{{[0-9]+}}+#0)
define dso_local void @volatile_loads(ptr noundef %dst, ptr noundef %src0) local_unnamed_addr {
entry:
  %0 = load volatile i32, ptr %src0, align 8
  %src1 = getelementptr i8, ptr %src0, i32 4
  %conv = zext i32 %0 to i64
  %1 = load volatile i32, ptr %src1, align 4
  %conv4 = zext i32 %1 to i64
  %shl = shl nuw i64 %conv4, 32
  %or = or disjoint i64 %shl, %conv
  store i64 %or, ptr %dst, align 1
  ret void
}

; CHECK-LABEL: volatile_stores:
; CHECK: memw(r{{[0-9]+}}+#0) = r{{[0-9]+}}
; CHECK: memw(r{{[0-9]+}}+#4) = r{{[0-9]+}}
; CHECK-NOT: memd(r{{[0-9]+}}+#0) = r{{[0-9]+}}
define dso_local void @volatile_stores(ptr noundef %dst0, i32 %a, i32 %b) local_unnamed_addr {
entry:
  store volatile i32 %a, ptr %dst0, align 8
  %dst1 = getelementptr i8, ptr %dst0, i32 4
  store volatile i32 %b, ptr %dst1, align 4
  ret void
}
