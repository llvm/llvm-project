; RUN: opt -passes=verify %s

define swifttailcc void @valid_attrs(ptr sret(i64) %ret, ptr byval(i8) %byval, ptr swiftself %self, ptr swiftasync %ctx) {
  musttail call swifttailcc void @valid_attrs(ptr sret(i64) %ret, ptr byval(i8) %byval, ptr swiftself %self, ptr swiftasync %ctx)
  ret void
}

define swifttailcc void @mismatch_parms() {
  musttail call swifttailcc void @valid_attrs(ptr sret(i64) undef, ptr byval(i8) undef, ptr swiftself undef, ptr swiftasync  undef)
  ret void
}
