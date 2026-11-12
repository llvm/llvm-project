int main(int argc, const char *argv[]) {
  struct MatrixData data = {0};
  data.section.origin.row = 1;
  data.section.origin.col = 2;
  data.section.size.row = 3;
  data.section.size.col = 4;
  data.stride = 5;

  return data.section.size.row;
}
