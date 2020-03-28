int patval;          // external symbol, will not be completely stripped
int pat(int in) {    // external symbol, will not be completely stripped
  if (patval == 0)
    patval = in;
  return patval;
}

static int fooval;  // static symbol, stripped
int foo() {         // external symbol, will not be completely stripped
  if (fooval == 0)
    fooval = 5;
  return fooval;
}

int bazval = 10;   // external symbol, will not be completely stripped
int baz () {       // external symbol, will not be completely stripped
  return foo() + bazval;
}

static int barval = 15; // static symbol, stripped
static int bar () {     // static symbol, stripped; __lldb_unnamed_symbol from func starts
  return baz() + barval;
}

int calculate ()   // external symbol, will not be completely stripped
{
  return bar();
}

