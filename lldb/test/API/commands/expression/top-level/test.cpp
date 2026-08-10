class MyClass
{
public:
  int memberResult()
  {
    return 1;
  }
  static int staticResult()
  {
    return 1;
  }
  int externResult();
};

// --

int MyClass::externResult()
{
  return 1;
}

// --

MyClass m;

// --

enum MyEnum {
  myEnumOne = 1,
  myEnumTwo,
  myEnumThree
};

// --

class AnotherClass
{
public:
    __attribute__ ((always_inline)) int complicatedFunction() 
    {
        struct {
            int i;
        } s = { 15 };
    
        int numbers[4] = { 2, 3, 4, 5 };
    
        for (signed char number: numbers)
        {
            s.i -= number;
        }
    
        return s.i;
    }
};

// --

class DiamondA
{
private:
  struct {
    int m_i;
  };
public:
  DiamondA(int i) : m_i(i) { }
  int accessor() { return m_i; }
};

// --

class DiamondB : public virtual DiamondA
{
public:
  DiamondB(int i) : DiamondA(i) { }
};

// --

class DiamondC : public virtual DiamondA
{
public:
  DiamondC(int i) : DiamondA(i) { }
};

// --

class DiamondD : public DiamondB, public DiamondC
{
public:
  DiamondD(int i) : DiamondA(i), DiamondB(i), DiamondC(i) { }
};

// --

int doTest()
{
    int accumulator = m.memberResult();
    accumulator += MyClass::staticResult();
    accumulator += m.externResult();
    accumulator += MyEnum::myEnumThree;
    accumulator += myEnumOne;
    accumulator += AnotherClass().complicatedFunction();
    accumulator += DiamondD(3).accessor();
    return accumulator;
}

// --
