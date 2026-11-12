
void test_mpz_export(char **out, char *rop, size_t *countp, int order,
                     size_t size, int endian, size_t nails, char *mpzstr) {
  mpz_t op;
  mpz_init(op);
  mpz_set_str(op, mpzstr, 10);
  // printf("%p,%p,%d,%zi,%d,%zi,%s\n", rop, countp, order, size, endian, nails,
  // mpzstr);
  *out = mpz_export(rop, countp, order, size, endian, nails, op);
}

void test_mpz_import(char *out, void *unused, size_t count, int order,
                     size_t size, int endian, size_t nails, char *mpzstr) {
  mpz_t op;
  mpz_t rop;
  mpz_init(op);
  mpz_init(rop);
  mpz_set_str(op, mpzstr, 10);
  char *data;

  // printf("%p,%p,%d,%zi,%d,%zi,%s\n", rop, countp, order, size, endian, nails,
  // mpzstr);
  data = mpz_export(NULL, &count, order, size, endian, nails, op);
  mpz_import(rop, count, order, size, endian, nails, data);
  int eq = mpz_cmpabs(op, rop);
  sprintf(out, "%2d:", eq);
  mpz_get_str(out + 3, 10, rop);
}
