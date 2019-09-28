#ifndef LLVM_DFA_FLOWSET_H
#define LLVM_DFA_FLOWSET_H
#include <iostream>

using namespace std;
namespace llvm{

template <typename T>
class FlowSet
{
    class Node;

public:
    FlowSet<T>() noexcept
    {
	//m_spRoot = nullptr;
    }

    class Iterator;

    Iterator begin()
    {
	return Iterator(m_spRoot);
    }
    
    Iterator end()
    {
	return Iterator(nullptr);
    }

    void push_back(T data);
    //void traverse();

    class Iterator
    {
    public:
	Iterator() noexcept:
	    m_pCurrentNode(m_spRoot){	}

	Iterator(const Node* pNode) noexcept:
	    m_pCurrentNode(pNode){  }

	Iterator& operator=(Node* pNode)
	{
	    this->m_pCurrentNode = pNode;
	    return *this;
	}

	Iterator& operator++()
	{
	    if(m_pCurrentNode)
		m_pCurrentNode = m_pCurrentNode->pNext;
	    return *this;
	}

	Iterator& operator++(int)
	{
	    Iterator it = *this;
	    ++*this;
	    return it;
	}

	Iterator& operator!=(const Iterator& it)
	{
	    return m_pCurrentNode != it.m_pCurrentNode;
	}

	int operator*()
	{
	    return m_pCurrentNode->data;
	}
    private:
	const Node* m_pCurrentNode;
    };

private:
    class Node
    {
	T data;
	Node* pNext;

	friend class FlowSet;
    };

    Node* getNode(T d)
    {
	Node* pTmp = new Node;
	pTmp->data = d;
	pTmp->pNext = nullptr;
	return pTmp;
    }
    Node*& getRoot()
    {
	return m_spRoot;
    }
    static Node* m_spRoot;
};

}   // end namepspace llvm

#endif	// LLVM_DFA_FLOWSET_H
