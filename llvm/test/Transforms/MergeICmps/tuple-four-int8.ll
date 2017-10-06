; RUN: opt -mergeicmps -S -o - %s | FileCheck %s

; This is a more involved test: clang generates this weird pattern for
; tuple<uint8_t, uint8_t, uint8_t, uint8_t>. Right now we skip the entry block
; (which defines the base pointer for other blocks) and the last one (which
; does not have the expected structure). Only middle blocks (bytes [1,2]) are
; merged.

%"class.std::tuple" = type { %"struct.std::_Tuple_impl" }
%"struct.std::_Tuple_impl" = type { %"struct.std::_Tuple_impl.0", %"struct.std::_Head_base.6" }
%"struct.std::_Tuple_impl.0" = type { %"struct.std::_Tuple_impl.1", %"struct.std::_Head_base.5" }
%"struct.std::_Tuple_impl.1" = type { %"struct.std::_Tuple_impl.2", %"struct.std::_Head_base.4" }
%"struct.std::_Tuple_impl.2" = type { %"struct.std::_Head_base" }
%"struct.std::_Head_base" = type { i8 }
%"struct.std::_Head_base.4" = type { i8 }
%"struct.std::_Head_base.5" = type { i8 }
%"struct.std::_Head_base.6" = type { i8 }

define zeroext i1 @opeq(
    %"class.std::tuple"* nocapture readonly dereferenceable(4) %a,
    %"class.std::tuple"* nocapture readonly dereferenceable(4) %b) local_unnamed_addr #1 {
entry:
  %a.base = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %a, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %a.elem3.addr = getelementptr inbounds i8, i8* %a.base, i64 3
  %0 = load i8, i8* %a.elem3.addr, align 1
  %b.base = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %b, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %b.elem3.addr = getelementptr inbounds i8, i8* %b.base, i64 3
  %1 = load i8, i8* %b.elem3.addr, align 1
  %cmp.elem3 = icmp eq i8 %0, %1
  br i1 %cmp.elem3, label %land.elem2, label %opeq.exit

land.elem2:
  %a.elem2.addr = getelementptr inbounds i8, i8* %a.base, i64 2
  %2 = load i8, i8* %a.elem2.addr, align 1
  %b.elem2.addr = getelementptr inbounds i8, i8* %b.base, i64 2
  %3 = load i8, i8* %b.elem2.addr, align 1
  %cmp.elem2 = icmp eq i8 %2, %3
  br i1 %cmp.elem2, label %land.elem1, label %opeq.exit

land.elem1:
  %a.elem1.addr = getelementptr inbounds i8, i8* %a.base, i64 1
  %4 = load i8, i8* %a.elem1.addr, align 1
  %b.elem1.addr = getelementptr inbounds i8, i8* %b.base, i64 1
  %5 = load i8, i8* %b.elem1.addr, align 1
  %cmp.elem1 = icmp eq i8 %4, %5
  br i1 %cmp.elem1, label %land.elem0, label %opeq.exit

land.elem0:
  %6 = load i8, i8* %a.base, align 1
  %7 = load i8, i8* %b.base, align 1
  %cmp.elem0 = icmp eq i8 %6, %7
  br label %opeq.exit

opeq.exit:
  %8 = phi i1 [ false, %entry ], [ false, %land.elem2 ], [ false, %land.elem1 ], [ %cmp.elem0, %land.elem0 ]
  ret i1 %8
; CHECK-LABEL: @opeq(
; The entry block is kept as is, but the next block is now the merged comparison
; block for bytes [1,2] or the block for the head.
; CHECK:     entry
; CHECK:     br i1 %cmp.elem3, label %land.elem{{[01]}}, label %opeq.exit
; The two 1 byte loads and compares at offset 1 are replaced with a single
; 2-byte memcmp.
; CHECK:     land.elem1
; CHECK:     @memcmp({{.*}}2)
; CHECK:     icmp eq {{.*}} 0
; In the end we have three blocks.
; CHECK: phi i1
; CHECK-SAME %entry
; CHECK-SAME %land.elem0
; CHECK-SAME %land.elem1
}

