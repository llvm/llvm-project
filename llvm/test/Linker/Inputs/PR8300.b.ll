%foo = type { [8 x i8] }
%bar = type { [9 x i8] }

@zed = alias void (%bar), ptr @xyz

define void @xyz(%foo %this) {
entry:
  ret void
}
