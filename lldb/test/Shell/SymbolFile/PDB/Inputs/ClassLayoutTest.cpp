// To avoid linking MSVC specific libs, we don't test virtual/override methods
// that needs vftable support in this file.

// Enum.
enum Enum { RED, GREEN, BLUE };
Enum EnumVar;

// Union.
union Union {
  short Row;
  unsigned short Col;
  int Line : 16; // Test named bitfield.
  short : 8;     // Unnamed bitfield symbol won't be generated in PDB.
  long Table;
};
Union UnionVar;

// Struct.
struct Struct;
typedef Struct StructTypedef;

struct Struct {
  bool A;
  unsigned char UCharVar;
  unsigned int UIntVar;
  long long LongLongVar;
  Enum EnumVar; // Test struct has UDT member.
  int array[10];
};
struct Struct StructVar;

struct _List; // Forward declaration.
struct Complex {
  struct _List *array[90];
  struct { // Test unnamed struct. MSVC treats it as `int x`
    int x;
  };
  union { // Test unnamed union. MSVC treats it as `int a; float b;`
    int a;
    float b;
  };
};
struct Complex c;

struct _List { // Test doubly linked list.
  struct _List *current;
  struct _List *previous;
  struct _List *next;
};
struct _List ListVar;

typedef struct {
  int a;
} UnnamedStruct; // Test unnamed typedef-ed struct.
UnnamedStruct UnnanmedVar;

// Class.
namespace MemberTest {
class Base {
public:
  Base() {}
  ~Base() {}

public:
  int Get() { return 0; }

protected:
  int a;
};
class Friend {
public:
  int f() { return 3; }
};
class Class : public Base { // Test base class.
  friend Friend;
  static int m_static; // Test static member variable.
public:
  Class() : m_public(), m_private(), m_protected() {}
  explicit Class(int a) { m_public = a; } // Test first reference of m_public.
  ~Class() {}

  static int StaticMemberFunc(int a, ...) {
    return 1;
  } // Test static member function.
  int Get() { return 1; }
  int f(Friend c) { return c.f(); }
  inline bool operator==(const Class &rhs) const // Test operator.
  {
    return (m_public == rhs.m_public);
  }

public:
  int m_public;
  struct Struct m_struct;

private:
  Union m_union;
  int m_private;

protected:
  friend class Friend;
  int m_protected;
};
} // namespace MemberTest

int main() {
  MemberTest::Base B1;
  B1.Get();
  MemberTest::Class::StaticMemberFunc(1, 10, 2);
  return 0;
}
