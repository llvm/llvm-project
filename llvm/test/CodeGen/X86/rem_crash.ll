; RUN: llc < %s -march=x86
; RUN: llc < %s -march=x86-64

define i8 @test_minsize_uu8(i8 %x) minsize optsize {
entry:
  %0 = udiv i8 %x, 10
  %1 = urem i8 %x, 10
  %res = add i8 %0, %1
  ret i8 %res
}

define i8 @test_minsize_ss8(i8 %x) minsize optsize {
entry:
  %0 = sdiv i8 %x, 10
  %1 = srem i8 %x, 10
  %res = add i8 %0, %1
  ret i8 %res
}

define i8 @test_minsize_us8(i8 %x) minsize optsize {
entry:
  %0 = udiv i8 %x, 10
  %1 = srem i8 %x, 10
  %res = add i8 %0, %1
  ret i8 %res
}

define i8 @test_minsize_su8(i8 %x) minsize optsize {
entry:
  %0 = sdiv i8 %x, 10
  %1 = urem i8 %x, 10
  %res = add i8 %0, %1
  ret i8 %res
}

define i16 @test_minsize_uu16(i16 %x) minsize optsize {
entry:
  %0 = udiv i16 %x, 10
  %1 = urem i16 %x, 10
  %res = add i16 %0, %1
  ret i16 %res
}

define i16 @test_minsize_ss16(i16 %x) minsize optsize {
entry:
  %0 = sdiv i16 %x, 10
  %1 = srem i16 %x, 10
  %res = add i16 %0, %1
  ret i16 %res
}

define i16 @test_minsize_us16(i16 %x) minsize optsize {
entry:
  %0 = udiv i16 %x, 10
  %1 = srem i16 %x, 10
  %res = add i16 %0, %1
  ret i16 %res
}

define i16 @test_minsize_su16(i16 %x) minsize optsize {
entry:
  %0 = sdiv i16 %x, 10
  %1 = urem i16 %x, 10
  %res = add i16 %0, %1
  ret i16 %res
}

define i32 @test_minsize_uu32(i32 %x) minsize optsize {
entry:
  %0 = udiv i32 %x, 10
  %1 = urem i32 %x, 10
  %res = add i32 %0, %1
  ret i32 %res
}

define i32 @test_minsize_ss32(i32 %x) minsize optsize {
entry:
  %0 = sdiv i32 %x, 10
  %1 = srem i32 %x, 10
  %res = add i32 %0, %1
  ret i32 %res
}

define i32 @test_minsize_us32(i32 %x) minsize optsize {
entry:
  %0 = udiv i32 %x, 10
  %1 = srem i32 %x, 10
  %res = add i32 %0, %1
  ret i32 %res
}

define i32 @test_minsize_su32(i32 %x) minsize optsize {
entry:
  %0 = sdiv i32 %x, 10
  %1 = urem i32 %x, 10
  %res = add i32 %0, %1
  ret i32 %res
}

define i64 @test_minsize_uu64(i64 %x) minsize optsize {
entry:
  %0 = udiv i64 %x, 10
  %1 = urem i64 %x, 10
  %res = add i64 %0, %1
  ret i64 %res
}

define i64 @test_minsize_ss64(i64 %x) minsize optsize {
entry:
  %0 = sdiv i64 %x, 10
  %1 = srem i64 %x, 10
  %res = add i64 %0, %1
  ret i64 %res
}

define i64 @test_minsize_us64(i64 %x) minsize optsize {
entry:
  %0 = udiv i64 %x, 10
  %1 = srem i64 %x, 10
  %res = add i64 %0, %1
  ret i64 %res
}

define i64 @test_minsize_su64(i64 %x) minsize optsize {
entry:
  %0 = sdiv i64 %x, 10
  %1 = urem i64 %x, 10
  %res = add i64 %0, %1
  ret i64 %res
}

define i8 @test_uu8(i8 %x) optsize {
entry:
  %0 = udiv i8 %x, 10
  %1 = urem i8 %x, 10
  %res = add i8 %0, %1
  ret i8 %res
}

define i8 @test_ss8(i8 %x) optsize {
entry:
  %0 = sdiv i8 %x, 10
  %1 = srem i8 %x, 10
  %res = add i8 %0, %1
  ret i8 %res
}

define i8 @test_us8(i8 %x) optsize {
entry:
  %0 = udiv i8 %x, 10
  %1 = srem i8 %x, 10
  %res = add i8 %0, %1
  ret i8 %res
}

define i8 @test_su8(i8 %x) optsize {
entry:
  %0 = sdiv i8 %x, 10
  %1 = urem i8 %x, 10
  %res = add i8 %0, %1
  ret i8 %res
}

define i16 @test_uu16(i16 %x) optsize {
entry:
  %0 = udiv i16 %x, 10
  %1 = urem i16 %x, 10
  %res = add i16 %0, %1
  ret i16 %res
}

define i16 @test_ss16(i16 %x) optsize {
entry:
  %0 = sdiv i16 %x, 10
  %1 = srem i16 %x, 10
  %res = add i16 %0, %1
  ret i16 %res
}

define i16 @test_us16(i16 %x) optsize {
entry:
  %0 = udiv i16 %x, 10
  %1 = srem i16 %x, 10
  %res = add i16 %0, %1
  ret i16 %res
}

define i16 @test_su16(i16 %x) optsize {
entry:
  %0 = sdiv i16 %x, 10
  %1 = urem i16 %x, 10
  %res = add i16 %0, %1
  ret i16 %res
}

define i32 @test_uu32(i32 %x) optsize {
entry:
  %0 = udiv i32 %x, 10
  %1 = urem i32 %x, 10
  %res = add i32 %0, %1
  ret i32 %res
}

define i32 @test_ss32(i32 %x) optsize {
entry:
  %0 = sdiv i32 %x, 10
  %1 = srem i32 %x, 10
  %res = add i32 %0, %1
  ret i32 %res
}

define i32 @test_us32(i32 %x) optsize {
entry:
  %0 = udiv i32 %x, 10
  %1 = srem i32 %x, 10
  %res = add i32 %0, %1
  ret i32 %res
}

define i32 @test_su32(i32 %x) optsize {
entry:
  %0 = sdiv i32 %x, 10
  %1 = urem i32 %x, 10
  %res = add i32 %0, %1
  ret i32 %res
}

define i64 @test_uu64(i64 %x) optsize {
entry:
  %0 = udiv i64 %x, 10
  %1 = urem i64 %x, 10
  %res = add i64 %0, %1
  ret i64 %res
}

define i64 @test_ss64(i64 %x) optsize {
entry:
  %0 = sdiv i64 %x, 10
  %1 = srem i64 %x, 10
  %res = add i64 %0, %1
  ret i64 %res
}

define i64 @test_us64(i64 %x) optsize {
entry:
  %0 = udiv i64 %x, 10
  %1 = srem i64 %x, 10
  %res = add i64 %0, %1
  ret i64 %res
}

define i64 @test_su64(i64 %x) optsize {
entry:
  %0 = sdiv i64 %x, 10
  %1 = urem i64 %x, 10
  %res = add i64 %0, %1
  ret i64 %res
}
