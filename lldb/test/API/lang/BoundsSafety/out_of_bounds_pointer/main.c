void bidi_check_known_type_size();
void bidi_check_unknown_type_size();
void idx_check_known_type_size();
void idx_check_unknown_type_size();

int main(void) {
  bidi_check_known_type_size();
  bidi_check_unknown_type_size();
  idx_check_known_type_size();
  idx_check_unknown_type_size();
  return 0;
}
