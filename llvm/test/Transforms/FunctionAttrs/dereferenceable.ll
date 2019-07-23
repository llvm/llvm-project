; RUN: opt -attributor --attributor-disable=false -S < %s | FileCheck %s --check-prefixes=ATTRIBUTOR


; TEST 1
; take mininimum of return values
;
define i32* @test1(i32* dereferenceable(4), double* dereferenceable(8), i1 zeroext) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(4) i32* @test1(i32* nonnull dereferenceable(4), double* nonnull dereferenceable(8), i1 zeroext)
  %4 = bitcast double* %1 to i32*
  %5 = select i1 %2, i32* %0, i32* %4
  ret i32* %5
}

; TEST 2
define i32* @test2(i32* dereferenceable_or_null(4), double* dereferenceable(8), i1 zeroext) local_unnamed_addr {
; ATTRIBUTOR: define dereferenceable_or_null(4) i32* @test2(i32* dereferenceable_or_null(4), double* nonnull dereferenceable(8), i1 zeroext)
  %4 = bitcast double* %1 to i32*
  %5 = select i1 %2, i32* %0, i32* %4
  ret i32* %5
}

; TEST 3
; GEP inbounds
define i32* @test3_1(i32* dereferenceable(8)) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(4) i32* @test3_1(i32* nonnull dereferenceable(8))
  %ret = getelementptr inbounds i32, i32* %0, i64 1
  ret i32* %ret
}

define i32* @test3_2(i32* dereferenceable_or_null(32)) local_unnamed_addr {
; FIXME: Argument should be mark dereferenceable because of GEP `inbounds`.
; ATTRIBUTOR: define nonnull dereferenceable(16) i32* @test3_2(i32* dereferenceable_or_null(32))
  %ret = getelementptr inbounds i32, i32* %0, i64 4
  ret i32* %ret
}

define i32* @test3_3(i32* dereferenceable(8), i32* dereferenceable(16), i1) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(4) i32* @test3_3(i32* nonnull dereferenceable(8), i32* nonnull dereferenceable(16), i1) local_unnamed_addr
  %ret1 = getelementptr inbounds i32, i32* %0, i64 1
  %ret2 = getelementptr inbounds i32, i32* %1, i64 2
  %ret = select i1 %2, i32* %ret1, i32* %ret2
  ret i32* %ret
}

; TEST 4
; Better than known in IR.

define dereferenceable(4) i32* @test4(i32* dereferenceable(8)) local_unnamed_addr {
; ATTRIBUTOR: define nonnull dereferenceable(8) i32* @test4(i32* nonnull returned dereferenceable(8))
  ret i32* %0
}

