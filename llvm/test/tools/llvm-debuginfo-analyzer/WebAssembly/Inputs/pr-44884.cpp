int bar(float Input) { return (int)Input; }

unsigned foo(char Param) {
  typedef int INT;
  INT Value = Param;
  {
    typedef float FLOAT;
    {
      FLOAT Added = Value + Param;
      Value = bar(Added);
    }
  }
  return Value + Param;
}
