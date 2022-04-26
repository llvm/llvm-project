; RUN: llc < %s
; PR9900

; NVPTX does not support zero sized type arg
; UNSUPPORTED: nvptx

%zero = type [0 x i8]
%foobar = type { i32, %zero }

define void @f(%foobar %arg) {
  %arg1 = extractvalue %foobar %arg, 0
  %arg2 = extractvalue %foobar %arg, 1
  call i32 @f2(%zero %arg2, i32 5, i32 42)
  ret void
}

define i32 @f2(%zero %x, i32 %y, i32 %z) {
  ret i32 %y
}

define void @f3(%zero %x, i32 %y) {
  call i32 @f2(%zero %x, i32 5, i32 %y)
  ret void
}

define void @f4(%zero %z) {
  insertvalue %foobar undef, %zero %z, 1
  ret void
}

define void @f5(%foobar %x) {
allocas:
  %y = extractvalue %foobar %x, 1
  br  label %b1

b1:
  %insert120 = insertvalue %foobar undef, %zero %y, 1
  ret void
}

define void @f6(%zero %x, %zero %y) {
b1:
  br i1 undef, label %end, label %b2

b2:
  br label %end

end:
  %z = phi %zero [ %y, %b1 ], [ %x, %b2 ]
  call void @f4(%zero %z)
  ret void
}

%zero2 = type {}

define i32 @g1(%zero2 %x, i32 %y, i32 %z) {
  ret i32 %y
}

define void @g2(%zero2 %x, i32 %y) {
  call i32 @g1(%zero2 %x, i32 5, i32 %y)
  ret void
}

%zero2r = type {%zero2}

define i32 @h1(%zero2r %x, i32 %y, i32 %z) {
  ret i32 %y
}

define void @h2(%zero2r %x, i32 %y) {
  call i32 @h1(%zero2r %x, i32 5, i32 %y)
  ret void
}

%foobar2 = type { i32, %zero2r }

define void @h3(%foobar2 %arg) {
  %arg1 = extractvalue %foobar2 %arg, 0
  %arg2 = extractvalue %foobar2 %arg, 1
  %arg21 = extractvalue %zero2r %arg2, 0
  call void @g2(%zero2 %arg21, i32 5)
  ret void
}
