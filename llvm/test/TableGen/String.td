// RUN: llvm-tblgen %s 

class x {
  string y = "missing terminating '\"' character";
}

