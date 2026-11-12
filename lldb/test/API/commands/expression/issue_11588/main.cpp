//
//  11588.cpp
//

#include <iostream>

class StgInfoTable {};

class StgHeader
{
private:
	StgInfoTable* info;
public:
	StgHeader()
	{
		info = new StgInfoTable();
	}
	~StgHeader()
	{
		delete info;
	}
};

class StgClosure
{
private:
	StgHeader header;
	StgClosure* payload[1];
public:
	StgClosure(bool make_payload = true)
	{
		if (make_payload)
			payload[0] = new StgClosure(false);
		else
			payload[0] = NULL;
	}
	~StgClosure()
	{
		if (payload[0])
			delete payload[0];
	}
};

typedef unsigned long long int ptr_type;

int main()
{
	StgClosure* r14_ = new StgClosure();
	r14_ = (StgClosure*)(((ptr_type)r14_ | 0x01)); // set the LSB to 1 for tagging
	ptr_type r14 = (ptr_type)r14_;
	int x = 0;
	x = 3;
	return (x-1); // Set breakpoint here.
}
