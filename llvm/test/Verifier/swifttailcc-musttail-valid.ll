; RUN: opt -verify %s

define swifttailcc void @valid_attrs(i64* sret(i64) %ret, i8* byval(i8) %byval, i8* swiftself %self, i8* swiftasync %ctx) {
  musttail call swifttailcc void @valid_attrs(i64* sret(i64) %ret, i8* byval(i8) %byval, i8* swiftself %self, i8* swiftasync %ctx)
  ret void
}

define swifttailcc void @mismatch_parms() {
  musttail call swifttailcc void @valid_attrs(i64* sret(i64) undef, i8* byval(i8) undef, i8* swiftself undef, i8* swiftasync  undef)
  ret void
}
