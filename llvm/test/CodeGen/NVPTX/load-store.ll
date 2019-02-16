; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s

; CHECK-LABEL: plain
define void @plain(i8* %a, i16* %b, i32* %c, i64* %d) local_unnamed_addr {
  ; CHECK: ld.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load i8, i8* %a
  %a.add = add i8 %a.load, 1
  ; CHECK: st.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store i8 %a.add, i8* %a

  ; CHECK: ld.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load i16, i16* %b
  %b.add = add i16 %b.load, 1
  ; CHECK: st.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store i16 %b.add, i16* %b

  ; CHECK: ld.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load i32, i32* %c
  %c.add = add i32 %c.load, 1
  ; CHECK: st.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store i32 %c.add, i32* %c

  ; CHECK: ld.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load i64, i64* %d
  %d.add = add i64 %d.load, 1
  ; CHECK: st.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store i64 %d.add, i64* %d

  ret void
}

; CHECK-LABEL: volatile
define void @volatile(i8* %a, i16* %b, i32* %c, i64* %d) local_unnamed_addr {
  ; CHECK: ld.volatile.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load volatile i8, i8* %a
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store volatile i8 %a.add, i8* %a

  ; CHECK: ld.volatile.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load volatile i16, i16* %b
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store volatile i16 %b.add, i16* %b

  ; CHECK: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load volatile i32, i32* %c
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store volatile i32 %c.add, i32* %c

  ; CHECK: ld.volatile.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load volatile i64, i64* %d
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store volatile i64 %d.add, i64* %d

  ret void
}

; CHECK-LABEL: monotonic
define void @monotonic(i8* %a, i16* %b, i32* %c, i64* %d) local_unnamed_addr {
  ; CHECK: ld.volatile.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %a.load = load atomic i8, i8* %a monotonic, align 1
  %a.add = add i8 %a.load, 1
  ; CHECK: st.volatile.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i8 %a.add, i8* %a monotonic, align 1

  ; CHECK: ld.volatile.u16 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
  %b.load = load atomic i16, i16* %b monotonic, align 2
  %b.add = add i16 %b.load, 1
  ; CHECK: st.volatile.u16 [%rd{{[0-9]+}}], %rs{{[0-9]+}}
  store atomic i16 %b.add, i16* %b monotonic, align 2

  ; CHECK: ld.volatile.u32 %r{{[0-9]+}}, [%rd{{[0-9]+}}]
  %c.load = load atomic i32, i32* %c monotonic, align 4
  %c.add = add i32 %c.load, 1
  ; CHECK: st.volatile.u32 [%rd{{[0-9]+}}], %r{{[0-9]+}}
  store atomic i32 %c.add, i32* %c monotonic, align 4

  ; CHECK: ld.volatile.u64 %rd{{[0-9]+}}, [%rd{{[0-9]+}}]
  %d.load = load atomic i64, i64* %d monotonic, align 8
  %d.add = add i64 %d.load, 1
  ; CHECK: st.volatile.u64 [%rd{{[0-9]+}}], %rd{{[0-9]+}}
  store atomic i64 %d.add, i64* %d monotonic, align 8

  ret void
}
