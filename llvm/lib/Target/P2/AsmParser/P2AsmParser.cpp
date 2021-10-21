//===-- P2AsmParser.cpp - Parse P2 assembly to MCInst instructions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "P2.h"
#include "MCTargetDesc/P2MCTargetDesc.h"
#include "MCTargetDesc/P2BaseInfo.h"
#include "P2RegisterInfo.h"
#include "TargetInfo/P2TargetInfo.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "p2-asm-parser"

namespace {
    class P2AssemblerOptions {
    public:
        P2AssemblerOptions():
            reorder(true), macro(true) {
        }

        bool isReorder() {return reorder;}
        void setReorder() {reorder = true;}
        void setNoreorder() {reorder = false;}

        bool isMacro() {return macro;}
        void setMacro() {macro = true;}
        void setNomacro() {macro = false;}

    private:
        bool reorder;
        bool macro;
    };
}

namespace {

    /// P2Operand - Instances of this class represent a parsed P2 machine
    /// instruction.
    class P2Operand : public MCParsedAsmOperand {

        enum KindTy {
            k_Immediate,
            k_Memory,
            k_Register,
            k_Token
        } Kind;

    public:
        P2Operand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

        struct PhysRegOp {
            unsigned RegNum; /// Register Number
        };
        struct ImmOp {
            const MCExpr *Val;
        };
        struct MemOp {
            unsigned Base;
            const MCExpr *Off;
        };

        union {
            StringRef Tok;
            struct PhysRegOp Reg;
            struct ImmOp Imm;
            struct MemOp Mem;
        };

        SMLoc StartLoc, EndLoc;

    public:
        void addRegOperands(MCInst &Inst, unsigned N) const {
            assert(N == 1 && "Invalid number of operands!");
            Inst.addOperand(MCOperand::createReg(getReg()));
        }

        void addExpr(MCInst &Inst, const MCExpr *Expr) const{
            // Add as immediate when possible.  Null MCExpr = 0.
            if (Expr == 0)
                Inst.addOperand(MCOperand::createImm(0));
            else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
                Inst.addOperand(MCOperand::createImm(CE->getValue()));
            else
                Inst.addOperand(MCOperand::createExpr(Expr));
        }

        void addImmOperands(MCInst &Inst, unsigned N) const {
            assert(N == 1 && "Invalid number of operands!");
            const MCExpr *Expr = getImm();
            addExpr(Inst,Expr);
        }

        void addMemOperands(MCInst &Inst, unsigned N) const {
            assert(N == 2 && "Invalid number of operands!");

            Inst.addOperand(MCOperand::createReg(getMemBase()));

            const MCExpr *Expr = getMemOff();
            addExpr(Inst,Expr);
        }

        bool isReg() const override { return Kind == k_Register; }
        bool isImm() const override { return Kind == k_Immediate; }
        bool isToken() const override { return Kind == k_Token; }
        bool isMem() const override { return Kind == k_Memory; }

        StringRef getToken() const {
            assert(Kind == k_Token && "Invalid access!");
            return Tok;
        }

        unsigned getReg() const override {
            assert((Kind == k_Register) && "Invalid access!");
            return Reg.RegNum;
        }

        const MCExpr *getImm() const {
            assert((Kind == k_Immediate) && "Invalid access!");
            return Imm.Val;
        }

        unsigned getMemBase() const {
            assert((Kind == k_Memory) && "Invalid access!");
            return Mem.Base;
        }

        const MCExpr *getMemOff() const {
            assert((Kind == k_Memory) && "Invalid access!");
            return Mem.Off;
        }

        static std::unique_ptr<P2Operand> CreateToken(StringRef Str, SMLoc S) {
            auto Op = std::make_unique<P2Operand>(k_Token);
            Op->Tok = Str;
            Op->StartLoc = S;
            Op->EndLoc = S;
            return Op;
        }

        /// Internal constructor for register kinds
        static std::unique_ptr<P2Operand> CreateReg(unsigned RegNum, SMLoc S, SMLoc E) {
            auto Op = std::make_unique<P2Operand>(k_Register);
            Op->Reg.RegNum = RegNum;
            Op->StartLoc = S;
            Op->EndLoc = E;
            return Op;
        }

        static std::unique_ptr<P2Operand> CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
            auto Op = std::make_unique<P2Operand>(k_Immediate);
            Op->Imm.Val = Val;
            Op->StartLoc = S;
            Op->EndLoc = E;
            return Op;
        }

        static std::unique_ptr<P2Operand> CreateMem(unsigned Base, const MCExpr *Off, SMLoc S, SMLoc E) {
            auto Op = std::make_unique<P2Operand>(k_Memory);
            Op->Mem.Base = Base;
            Op->Mem.Off = Off;
            Op->StartLoc = S;
            Op->EndLoc = E;
            return Op;
        }

        /// getStartLoc - Get the location of the first token of this operand.
        SMLoc getStartLoc() const override { return StartLoc; }
        /// getEndLoc - Get the location of the last token of this operand.
        SMLoc getEndLoc() const override { return EndLoc; }

        void print(raw_ostream &OS) const override {
            switch (Kind) {
            case k_Immediate:
                OS << "Imm<";
                OS << *Imm.Val;
                OS << ">";
                break;
            case k_Memory:
                OS << "Mem<";
                OS << Mem.Base;
                OS << ", ";
                OS << *Mem.Off;
                OS << ">";
                break;
            case k_Register:
                OS << "Register<" << Reg.RegNum << ">";
                break;
            case k_Token:
                OS << Tok;
                break;
            }
        }
    };

    class P2AsmParser : public MCTargetAsmParser {
        MCAsmParser &Parser;
        P2AssemblerOptions Options;

    #define GET_ASSEMBLER_HEADER
    #include "P2GenAsmMatcher.inc"

        bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode, OperandVector &Operands, MCStreamer &Out,
                                        uint64_t &ErrorInfo, bool MatchingInlineAsm) override;
        bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
        bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc, OperandVector &Operands) override;
        bool ParseDirective(AsmToken DirectiveID) override;
        OperandMatchResultTy tryParseRegister(unsigned &reg_no, SMLoc &start, SMLoc &end) override;

        bool parseOperand(OperandVector &Operands, StringRef Mnemonic);
        //bool parseConditionOperand(OperandVector &Operands, StringRef Mnemonic);
        //bool parseEffectOperand(OperandVector &Operands, StringRef Mnemonic);
        int parseRegister(StringRef Mnemonic);
        bool tryParseRegisterOperand(OperandVector &Operands, StringRef Mnemonic);
        int matchRegisterName(StringRef Symbol);
        int matchRegisterByNumber(unsigned RegNum, StringRef Mnemonic);
        unsigned getReg(int RC,int RegNo);

    public:
        P2AsmParser(const MCSubtargetInfo &sti, MCAsmParser &parser, const MCInstrInfo &MII, const MCTargetOptions &Options)
                    : MCTargetAsmParser(Options, sti, MII), Parser(parser) {
            // Initialize the set of available features.
            setAvailableFeatures(ComputeAvailableFeatures(getSTI().getFeatureBits()));
        }

        MCAsmParser &getParser() const { return Parser; }
        MCAsmLexer &getLexer() const { return Parser.getLexer(); }

        std::unique_ptr<P2Operand> defaultEffectOperands() {
            const MCConstantExpr *eff_expr = MCConstantExpr::create(0, getContext());
            return P2Operand::CreateImm(eff_expr, SMLoc(), SMLoc());
        };
    };
}

namespace {


}

void printP2Operands(OperandVector &Operands) {
    for (size_t i = 0; i < Operands.size(); i++) {
        P2Operand* op = static_cast<P2Operand*>(&*Operands[i]);
        assert(op != nullptr);
        LLVM_DEBUG(dbgs() << " " << *op);
    }
    LLVM_DEBUG(dbgs() << "\n");
}

/*
implement virtual functions
*/
bool P2AsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode, OperandVector &Operands,
                                            MCStreamer &Out, uint64_t &ErrorInfo, bool MatchingInlineAsm) {

    LLVM_DEBUG(errs() << "match and emit instruction\n");
    printP2Operands(Operands);
    MCInst Inst;
    unsigned MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm);

    switch (MatchResult) {
        default:
            break;
        case Match_Success: {
            Inst.setLoc(IDLoc);
            LLVM_DEBUG(Inst.dump());
            Out.emitInstruction(Inst, getSTI());
            return false;
        }

        case Match_MissingFeature:
            Error(IDLoc, "instruction requires a CPU feature not currently enabled");
            return true;
        case Match_InvalidOperand: {
            SMLoc ErrorLoc = IDLoc;
            if (ErrorInfo != ~0U) {
                if (ErrorInfo >= Operands.size())
                    return Error(IDLoc, "too few operands for instruction");

                ErrorLoc = ((P2Operand &)*Operands[ErrorInfo]).getStartLoc();
                if (ErrorLoc == SMLoc()) ErrorLoc = IDLoc;
            }

            LLVM_DEBUG(errs() << "error: " << ErrorInfo << "\n");

            return Error(ErrorLoc, "invalid operand for instruction");
        }
        case Match_MnemonicFail:
            return Error(IDLoc, "invalid instruction");
    }

    return true;
}

OperandMatchResultTy P2AsmParser::tryParseRegister(unsigned &reg_no, SMLoc &start, SMLoc &end) {
    start = Parser.getTok().getLoc();
    reg_no = parseRegister("");
    end = Parser.getTok().getLoc();

    if (reg_no == (unsigned)-1)
        return MatchOperand_NoMatch;

    return MatchOperand_Success;
}

bool P2AsmParser::ParseRegister(unsigned &reg_no, SMLoc &start, SMLoc &end) {
    start = Parser.getTok().getLoc();
    reg_no = parseRegister("");
    end = Parser.getTok().getLoc();
    return (reg_no == (unsigned)-1);
}

/*
 * Every assembly instruction has the following format:
 *
 * [cond code] <inst mnemonic> [0-3 operands, comma list] [wc/wz/wcz].
 *
 * so the parse this, we will start by lexing the first token and seeing what it is. If it's a conditional code, save an MCConstantExpr
 * and move on. Then, assume the next token is the instruction. If we didn't find a condition code, assume the token is an instruction.
 *
 * then, loop over the comma separated list of operands and call parseOperand() for each token.
 *
 * at the end of the comma separated list, assume the next token (if one exists) is the effect flag. if it's not, error out.
 *
 */
bool P2AsmParser::ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc, OperandVector &Operands) {

    LLVM_DEBUG(errs() << "=== Parse Instruction ===\n");

    LLVM_DEBUG(errs() << "name: \"" << Name << "\"\n");

    SMLoc cond_loc = getLexer().getLoc();
    SMLoc effect_loc;

    StringRef condition = "";

    // check if it's a condition string, and if it is, continue and get the instruction mnemonic
    if (P2::cond_string_map.find(Name) != P2::cond_string_map.end()) {
        condition = Name;
        NameLoc = getLexer().getLoc();
        Name = getLexer().getTok().getString();
        Parser.Lex(); // eat the token

        LLVM_DEBUG(errs() << "got a condition, next token is " << Name << " of size " << Name.size() << "\n");
    }

    // append the condition code first
    SMLoc e = SMLoc::getFromPointer(cond_loc.getPointer() - 1);
    const MCConstantExpr *cond_expr = MCConstantExpr::create(P2::cond_string_map[condition], getContext());
    Operands.push_back(P2Operand::CreateImm(cond_expr, cond_loc, e));

    Operands.push_back(P2Operand::CreateToken(Name, NameLoc));

    // if there are operands, parse them
    if (getLexer().isNot(AsmToken::EndOfStatement)) {

        // get the first operand
        if (parseOperand(Operands, Name)) {
            SMLoc Loc = getLexer().getLoc();
            Parser.eatToEndOfStatement();
            return Error(Loc, "unexpected token in argument list");
        }

        // get the remaining operands
        while (getLexer().is(AsmToken::Comma)) {
           Parser.Lex();  // Eat the comma

           LLVM_DEBUG(errs() << "got another operand\n");

            // Parse and remember the operand.
            if (parseOperand(Operands, Name)) {
                SMLoc Loc = getLexer().getLoc();
                Parser.eatToEndOfStatement();
                return Error(Loc, "unexpected token in operand list");
            }
        }
    }

    effect_loc = getLexer().getLoc();
    // if there's another token, see if it's the effect flag.
    StringRef effect_flag = "";
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
        effect_flag = getLexer().getTok().getString();
        effect_loc = getLexer().getLoc();
        Parser.Lex(); // eat the effect flag

        if (P2::effect_string_map.find(effect_flag) == P2::effect_string_map.end()) {
            SMLoc Loc = getLexer().getLoc();
            Parser.eatToEndOfStatement();
            return Error(Loc, "unexpected token for effect flag");
        }

        LLVM_DEBUG(errs() << "got effect flag\n");
        e = SMLoc::getFromPointer(effect_loc.getPointer() - 1);
        const MCConstantExpr *eff_expr = MCConstantExpr::create(P2::effect_string_map[effect_flag], getContext());
        Operands.push_back(P2Operand::CreateImm(eff_expr, effect_loc, e));
    }

    // if we still haven't reached the end of the statement, error out.
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
        SMLoc Loc = getLexer().getLoc();
        Parser.eatToEndOfStatement();
        LLVM_DEBUG(errs() << "haven't found the end of the statement\n");
        return Error(Loc, "unexpected token after effect flags");
    }

    Parser.Lex(); // Consume the EndOfStatement

    LLVM_DEBUG(errs() << "=== End Parse Instruction ===\n");
    return false;
}

bool P2AsmParser::ParseDirective(llvm::AsmToken DirectiveID) {
    LLVM_DEBUG(errs() << "Parse directive: " << DirectiveID.getString() << "\n");
    return true;
}

/*
support functions
*/
unsigned P2AsmParser::getReg(int RC,int RegNo) {
    return *(getContext().getRegisterInfo()->getRegClass(RC).begin() + RegNo);
}

int P2AsmParser::parseRegister(StringRef Mnemonic) {
    const AsmToken &Tok = Parser.getTok();
    int RegNum = -1;

    if (Tok.is(AsmToken::Identifier)) {
        std::string lowerCase = Tok.getString().lower();
        RegNum = matchRegisterName(lowerCase);
    } else if (Tok.is(AsmToken::Integer)) {
        RegNum = matchRegisterByNumber(static_cast<unsigned>(Tok.getIntVal()), Mnemonic.lower());
    } else {
        return RegNum;  //error
    }

    return RegNum;
}

int P2AsmParser::matchRegisterByNumber(unsigned RegNum, StringRef Mnemonic) {
    if (RegNum > 15)
        return -1;

    return getReg(P2::P2GPRRegClassID, RegNum);
}

int P2AsmParser::matchRegisterName(StringRef Name) {

    int reg = StringSwitch<unsigned>(Name)
            .Case("r0",     P2::R0)
            .Case("r1",     P2::R1)
            .Case("r2",     P2::R2)
            .Case("r3",     P2::R3)
            .Case("r4",     P2::R4)
            .Case("r5",     P2::R5)
            .Case("r6",     P2::R6)
            .Case("r7",     P2::R7)
            .Case("r8",     P2::R8)
            .Case("r9",     P2::R9)
            .Case("r10",    P2::R10)
            .Case("r11",    P2::R11)
            .Case("r12",    P2::R12)
            .Case("r13",    P2::R13)
            .Case("r14",    P2::R14)
            .Case("r15",    P2::R15)
            .Case("r16",    P2::R16)
            .Case("r17",    P2::R17)
            .Case("r18",    P2::R18)
            .Case("r19",    P2::R19)
            .Case("r20",    P2::R20)
            .Case("r21",    P2::R21)
            .Case("r22",    P2::R22)
            .Case("r23",    P2::R23)
            .Case("r24",    P2::R24)
            .Case("r25",    P2::R25)
            .Case("r26",    P2::R26)
            .Case("r27",    P2::R27)
            .Case("r28",    P2::R28)
            .Case("r29",    P2::R29)
            .Case("r30",    P2::R30)
            .Case("r31",    P2::R31)
            .Case("ijmp3",  P2::IJMP3)
            .Case("iret3",  P2::IRET3)
            .Case("ijmp2",  P2::IJMP2)
            .Case("iret2",  P2::IRET2)
            .Case("ijmp1",  P2::IJMP1)
            .Case("iret1",  P2::IRET1)
            .Case("pa",     P2::PA)
            .Case("pb",     P2::PB)
            .Case("ptra",   P2::PTRA)
            .Case("ptrb",   P2::PTRB)
            .Case("dira",   P2::DIRA)
            .Case("dirb",   P2::DIRB)
            .Case("outa",   P2::OUTA)
            .Case("outb",   P2::OUTB)
            .Case("ina",    P2::INA)
            .Case("inb",    P2::INB)
            .Default(-1);

    return reg;
}

bool P2AsmParser::tryParseRegisterOperand(OperandVector &Operands, StringRef Mnemonic) {

    SMLoc S = Parser.getTok().getLoc();
    SMLoc E = Parser.getTok().getEndLoc();
    int RegNo = parseRegister(Mnemonic);

    if (RegNo == -1) {
        return true;
    }

    Operands.push_back(P2Operand::CreateReg(RegNo, S, E));
    Parser.Lex(); // Eat register token.
    return false;
}

bool P2AsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {

    LLVM_DEBUG(dbgs() << "Generic Operand Parser\n");

    AsmToken::TokenKind tok_kind = getLexer().getKind();

    switch (tok_kind) {
        default:
            Error(Parser.getTok().getLoc(), "unexpected token in operand");
            return true;
        case AsmToken::Dollar: {
            LLVM_DEBUG(errs() << "operand token is a $\n");
            // parse register
            SMLoc S = Parser.getTok().getLoc();
            Parser.Lex(); // Eat dollar token.

            // parse register operand
            if (!tryParseRegisterOperand(Operands, Mnemonic)) {
                return false; // success
            }

            // maybe it is a symbol reference
            StringRef Identifier;
            if (Parser.parseIdentifier(Identifier)) {
                return true; // fail
            }

            SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
            MCSymbol *Sym = getContext().getOrCreateSymbol("$" + Identifier);

            // Otherwise create a symbol ref.
            const MCExpr *Res = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None,getContext());
            Operands.push_back(P2Operand::CreateImm(Res, S, E));
            return false;
        }

        case AsmToken::Hash: {
            LLVM_DEBUG(errs() << "operand token is a #\n");
            // is an immediate expression, so first create the token for the #
            SMLoc S = Parser.getTok().getLoc();
            Parser.Lex(); // eat the pound sign
            Operands.push_back(P2Operand::CreateToken("#", S));

            const MCExpr *IdVal;
            S = Parser.getTok().getLoc();
            if (getParser().parseExpression(IdVal))
                return true;

            SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
            Operands.push_back(P2Operand::CreateImm(IdVal, S, E));
            return false;
        }
    }
    return true;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeP2AsmParser() {
    RegisterMCAsmParser<P2AsmParser> X(getTheP2Target());
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "P2GenAsmMatcher.inc"