#ifndef OBS_LEXER_H
#define OBS_LEXER_H

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace obs {

  struct Location {
    std::shared_ptr<std::string> file;
    int line;
    int col;
  };
  
  enum Token : int {
    tok_semicolon = ';',
    tok_parenthese_open = '(',
    tok_parenthese_close = ')',
    tok_bracket_open = '{',
    tok_bracket_close = '}',
    tok_sbracket_open = '[',
    tok_sbracket_close = ']',
    tok_eof = - 1,
    tok_return = -2,
    tok_var = -3,
    tok_def = -4,

    tok_identifier = -5,
    tok_number = -6,
  };

class Lexer {
public:
  Lexer(std::string filename) : lastLocation( {std::make_shared<std::string>(std::move(filename)), 0, 0} ) { }
  virtual ~Lexer() = default;

  Token getCurToken() {
    return curTok;
  }

  Token getNextToken() {
    return curTok = getTok();
  }

  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  llvm::StringRef getId() {
    assert(curTok == tok_identifier);
    return identifierStr;
  }

  double getValue() {
    assert(curTok == tok_number);
    return numVal;
  }


  Token getTok() {
    while(isspace(lastChar)) {
      lastChar = Token(getNextChar());
    }

    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    if (isalpha(lastChar)) {
      identifierStr = (char)lastChar;
      while(isalnum(lastChar = Token(getNextChar())) || lastChar == '_') {
        identifierStr += (char)lastChar;
      }
      if (identifierStr == "return") {
        return tok_return;
      }
      if (identifierStr == "def") {
        return tok_def;
      }
      if (identifierStr == "var") {
        return tok_var;
      }
      return tok_identifier;
    }  

    //Number: [0-9] +
    if (isdigit(lastChar) || lastChar == '.') {
      std::string numStr;
      do {
        numStr += lastChar;
        lastChar = Token(getNextChar());
      } while(isdigit(lastChar) || lastChar == '.');

      numVal = strtod(numStr.c_str(), nullptr);
      return tok_number;
    }

    if (lastChar == '#') {
      do {
        lastChar = Token(getNextChar());
      } while( lastChar != EOF && lastChar != '\n' && lastChar != '\r');

      if (lastChar != EOF) {
        return getTok();
      }
    }
    if (lastChar == EOF) {
      return tok_eof;
    }

    Token thisChar = Token(lastChar);
    lastChar = Token(getNextChar());
    return thisChar;
  }

  Location getLastLocation() {
    return lastLocation;
  }

  int getLine(){
    return curLineNum;
  }

  int getCol() {
    return curCol;
  }



private:

  virtual llvm::StringRef readNextLine() = 0;

  int getNextChar() {
    if (curLineBuffer.empty()) {
      return EOF;
    }
    ++curCol;
    auto nextChar = curLineBuffer.front();
    curLineBuffer = curLineBuffer.drop_front();
    if (curLineBuffer.empty()) {
      curLineBuffer = readNextLine();
    }
    if (nextChar == '\n') {
      ++curLineNum;
      curCol = 0;
    }
    return nextChar;
  }
  //Private member variables.
  Location lastLocation;
  Token curTok = tok_eof;
  Token lastChar = Token(' ');
  llvm::StringRef curLineBuffer = "\n";
  int curCol = 0;
  int curLineNum = 0;

  std::string identifierStr;
  double numVal = 0;
};

class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char *begin, const char *end, std::string filename) : Lexer(std::move(filename)), current(begin), end(end) {}
private:
  const char* current, *end;
  
  llvm::StringRef readNextLine() override {
    auto *begin = current;
    while (current <= end && *current && *current !='\n') {
      ++current;
    };
    if (current <= end && *current ) {
      ++current;
    };
    llvm::StringRef result{begin, static_cast<size_t>(current - begin)};
    return result;
  };
};

}



#endif //OBS_LEXER_H