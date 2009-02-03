// RUN: not tblgen %s 2> /dev/null

def {
  bits<2> X = 5;  // bitfield is too small, reject
}
