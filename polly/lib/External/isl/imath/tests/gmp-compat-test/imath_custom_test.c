
void test_mpz_export(char **out, char *rop, size_t *countp, int order,
                     size_t size, int endian, size_t nails, char *mpzstr) {
  impz_t op;
  impz_init(op);
  impz_set_str(op, mpzstr, 10);
  // printf("%p,%p,%d,%zi,%d,%zi,%s\n", rop, countp, order, size, endian, nails,
  // mpzstr);
  *out = impz_export(rop, countp, order, size, endian, nails, op);
}

void test_mpz_import(char *out, void *unused, size_t count, int order,
                     size_t size, int endian, size_t nails, char *mpzstr) {
  impz_t op;
  impz_t rop;
  impz_init(op);
  impz_init(rop);
  impz_set_str(op, mpzstr, 10);
  char *data;

  // printf("%p,%p,%d,%zi,%d,%zi,%s\n", rop, countp, order, size, endian, nails,
  // mpzstr);
  data = impz_export(NULL, &count, order, size, endian, nails, op);
  impz_import(rop, count, order, size, endian, nails, data);
  int eq = impz_cmpabs(op, rop);
  sprintf(out, "%2d:", eq);
  impz_get_str(out + 3, 10, rop);
}
