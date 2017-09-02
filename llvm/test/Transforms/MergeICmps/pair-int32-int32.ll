; RUN: opt -mergeicmps -S -o - %s | FileCheck %s

%"struct.std::pair" = type { i32, i32 }

define zeroext i1 @opeq1(
    %"struct.std::pair"* nocapture readonly dereferenceable(8) %a,
    %"struct.std::pair"* nocapture readonly dereferenceable(8) %b) local_unnamed_addr #0 {
entry:
  %first.i = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %a, i64 0, i32 0
  %0 = load i32, i32* %first.i, align 4
  %first1.i = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %b, i64 0, i32 0
  %1 = load i32, i32* %first1.i, align 4
  %cmp.i = icmp eq i32 %0, %1
  br i1 %cmp.i, label %land.rhs.i, label %opeq1.exit

land.rhs.i:
  %second.i = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %a, i64 0, i32 1
  %2 = load i32, i32* %second.i, align 4
  %second2.i = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %b, i64 0, i32 1
  %3 = load i32, i32* %second2.i, align 4
  %cmp3.i = icmp eq i32 %2, %3
  br label %opeq1.exit

opeq1.exit:
  %4 = phi i1 [ false, %entry ], [ %cmp3.i, %land.rhs.i ]
  ret i1 %4
; CHECK-LABEL: @opeq1(
; The entry block with zero-offset GEPs is kept, loads are removed.
; CHECK: entry
; CHECK:     getelementptr {{.*}} i32 0
; CHECK-NOT: load
; CHECK:     getelementptr {{.*}} i32 0
; CHECK-NOT: load
; The two 4 byte loads and compares are replaced with a single 8-byte memcmp.
; CHECK:     @memcmp({{.*}}8)
; CHECK:     icmp eq {{.*}} 0
; The branch is now a direct branch; the other block has been removed.
; CHECK:     br label %opeq1.exit
; CHECK-NOT: br
; The phi is updated.
; CHECK:      phi i1 [ %{{[^,]*}}, %entry ]
; CHECK-NEXT: ret
}

; Same as above, but the two blocks are in inverse order.
define zeroext i1 @opeq1_inverse(
    %"struct.std::pair"* nocapture readonly dereferenceable(8) %a,
    %"struct.std::pair"* nocapture readonly dereferenceable(8) %b) local_unnamed_addr #0 {
entry:
  %first.i = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %a, i64 0, i32 1
  %0 = load i32, i32* %first.i, align 4
  %first1.i = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %b, i64 0, i32 1
  %1 = load i32, i32* %first1.i, align 4
  %cmp.i = icmp eq i32 %0, %1
  br i1 %cmp.i, label %land.rhs.i, label %opeq1.exit

land.rhs.i:
  %second.i = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %a, i64 0, i32 0
  %2 = load i32, i32* %second.i, align 4
  %second2.i = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %b, i64 0, i32 0
  %3 = load i32, i32* %second2.i, align 4
  %cmp3.i = icmp eq i32 %2, %3
  br label %opeq1.exit

opeq1.exit:
  %4 = phi i1 [ false, %entry ], [ %cmp3.i, %land.rhs.i ]
  ret i1 %4
; CHECK-LABEL: @opeq1_inverse(
; The second block with zero-offset GEPs is kept, loads are removed.
; CHECK: land.rhs.i
; CHECK:     getelementptr {{.*}} i32 0
; CHECK-NOT: load
; CHECK:     getelementptr {{.*}} i32 0
; CHECK-NOT: load
; The two 4 byte loads and compares are replaced with a single 8-byte memcmp.
; CHECK:     @memcmp({{.*}}8)
; CHECK:     icmp eq {{.*}} 0
; The branch is now a direct branch; the other block has been removed.
; CHECK:     br label %opeq1.exit
; CHECK-NOT: br
; The phi is updated.
; CHECK:      phi i1 [ %{{[^,]*}}, %land.rhs.i ]
; CHECK-NEXT: ret
}



