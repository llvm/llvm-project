int _ID = 0;

class Foo {
	public:
		Foo *next;
		int ID;
	
	Foo () : next(0), ID(++_ID) {}
};

int evalFoo(Foo& foo)
{
	return foo.ID; // Set break point at this line.
}

int main() {
	Foo f;
	f.next = &f;
	return evalFoo(f);
}

