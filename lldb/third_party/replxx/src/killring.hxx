#ifndef REPLXX_KILLRING_HXX_INCLUDED
#define REPLXX_KILLRING_HXX_INCLUDED 1

#include <vector>

#include "unicodestring.hxx"

namespace replxx {

class KillRing {
	static const int capacity = 10;
	int size;
	int index;
	char indexToSlot[10];
	std::vector<UnicodeString> theRing;

public:
	enum action { actionOther, actionKill, actionYank };
	action lastAction;

	KillRing()
		: size(0)
		, index(0)
		, lastAction(actionOther) {
		theRing.reserve(capacity);
	}

	void kill(const char32_t* text, int textLen, bool forward) {
		if (textLen == 0) {
			return;
		}
		UnicodeString killedText(text, textLen);
		if (lastAction == actionKill && size > 0) {
			int slot = indexToSlot[0];
			int currentLen = static_cast<int>(theRing[slot].length());
			UnicodeString temp;
			if ( forward ) {
				temp.append( theRing[slot].get(), currentLen ).append( killedText.get(), textLen );
			} else {
				temp.append( killedText.get(), textLen ).append( theRing[slot].get(), currentLen );
			}
			theRing[slot] = temp;
		} else {
			if (size < capacity) {
				if (size > 0) {
					memmove(&indexToSlot[1], &indexToSlot[0], size);
				}
				indexToSlot[0] = size;
				size++;
				theRing.push_back(killedText);
			} else {
				int slot = indexToSlot[capacity - 1];
				theRing[slot] = killedText;
				memmove(&indexToSlot[1], &indexToSlot[0], capacity - 1);
				indexToSlot[0] = slot;
			}
			index = 0;
		}
	}

	UnicodeString* yank() { return (size > 0) ? &theRing[indexToSlot[index]] : 0; }

	UnicodeString* yankPop() {
		if (size == 0) {
			return 0;
		}
		++index;
		if (index == size) {
			index = 0;
		}
		return &theRing[indexToSlot[index]];
	}
};

}

#endif

