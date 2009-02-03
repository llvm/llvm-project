// RUN: tblgen %s | grep -- 4294901760

def X {
  int Y = 0xFFFF0000;
}
