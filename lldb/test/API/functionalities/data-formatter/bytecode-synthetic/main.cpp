struct Account {
  char *username;
  char *password;
};

void stop() {}

int main(int argc, char **argv) {
  struct Account acc;
  acc.username = "jake";
  acc.password = "popcorn";
  stop(); // break here
  return 0;
}

__attribute__((used, section("__DATA_CONST,__lldbformatters"))) unsigned char
    _Account_synthetic[] =
        "\x01"                      // version
        "\x15"                      // remaining record size
        "\x07"                      // type name size
        "Account"                   // type name
        "\x00"                      // flags
        "\x02"                      // sig_get_num_children
        "\x02"                      // program size
        "\x20\x01"                  // `return 1`
        "\x04"                      // sig_get_child_at_index
        "\x06"                      // program size
        "\x02\x20\x00\x23\x11\x60"; // `return self.valobj.GetChildAtIndex(0)`
