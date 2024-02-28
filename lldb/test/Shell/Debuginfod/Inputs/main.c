// A script to (re)create the .yaml files is in 'make-inputs'. If you make changes
// you'll need to update the .note.gnu.buildid values in the tests, as the cache names

int func(int argc, const char **argv) {
  return (argc + 1) * (argv[argc][0] + 2);
}

int main(int argc, const char *argv[]) {
  return func(0, argv);
}
