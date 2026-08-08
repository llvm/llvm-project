struct VBase1 {
  short member = 1;
};
struct VBase2 {
  short member = 2;
};

struct Padding3 {
  short member = 3;
};
struct Padding4 {
  short member = 4;
};
struct Padding5 {
  short member = 5;
};

struct User : public Padding3, public virtual VBase1, public virtual VBase2 {
  short member = 6;
};

struct UserUser : public Padding4, public User, public Padding5 {
  short member = 7;
};

int main() {
  UserUser useruser;

  return 0; // break here
}
