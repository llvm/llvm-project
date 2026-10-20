; RUN: not opt -disable-output -passes='forceattrs' -forceattrs-csv-path="%S/CannotOpenFile.csv"  %s 2>&1 | FileCheck -DMSG=%errc_ENOENT %s

; CHECK: error: cannot open CSV file: [[MSG]]
define void @first_function() {
  ret void
}
