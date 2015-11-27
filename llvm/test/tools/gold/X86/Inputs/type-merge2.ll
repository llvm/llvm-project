%zed = type { i16 }
define void @bar(%zed* %this)  {
  store %zed* %this, %zed** null
  ret void
}
