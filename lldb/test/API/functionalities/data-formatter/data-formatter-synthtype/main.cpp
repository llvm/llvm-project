class myInt {
    private: int theValue;
    public: myInt() : theValue(0) {}
    public: myInt(int _x) : theValue(_x) {}
    int val() { return theValue; }
};

class myArray {
public:
    int array[16];
};

class hasAnInt {
    public:
        myInt theInt;
        hasAnInt() : theInt(42) {}  
};

myInt operator + (myInt x, myInt y) { return myInt(x.val() + y.val()); }

int main() {
    myInt x{3};
    myInt y{4};
    myInt z {x+y};
    hasAnInt hi;
    myArray ma;

    return z.val(); // break here
}
