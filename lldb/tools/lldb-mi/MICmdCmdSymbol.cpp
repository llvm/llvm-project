//===-- MICmdCmdSymbol.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Overview:    CMICmdCmdSymbolListLines     implementation.

// Third Party Headers:
#include "lldb/API/SBCommandInterpreter.h"

// In-house headers:
#include "MICmdArgValFile.h"
#include "MICmdCmdSymbol.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueList.h"
#include "MICmnMIValueTuple.h"
#include "MIUtilParse.h"

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdSymbolListLines constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdSymbolListLines::CMICmdCmdSymbolListLines()
    : m_constStrArgNameFile("file")
{
    // Command factory matches this name with that received from the stdin stream
    m_strMiCmd = "symbol-list-lines";

    // Required by the CMICmdFactory when registering *this command
    m_pSelfCreatorFn = &CMICmdCmdSymbolListLines::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details: CMICmdCmdSymbolListLines destructor.
// Type:    Overrideable.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CMICmdCmdSymbolListLines::~CMICmdCmdSymbolListLines()
{
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The parses the command line options
//          arguments to extract values for each of those arguments.
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdSymbolListLines::ParseArgs()
{
    m_setCmdArgs.Add(new CMICmdArgValFile(m_constStrArgNameFile, true, true));
    return ParseValidateCmdOptions();
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The command does work in this function.
//          The command is likely to communicate with the LLDB SBDebugger in here.
//          Synopsis: -symbol-list-lines file
//          Ref: http://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI-Symbol-Query.html#GDB_002fMI-Symbol-Query
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdSymbolListLines::Execute()
{
    CMICMDBASE_GETOPTION(pArgFile, File, m_constStrArgNameFile);

    const CMIUtilString &strFilePath(pArgFile->GetValue());
    const CMIUtilString strCmd(CMIUtilString::Format("source info --file \"%s\"", strFilePath.AddSlashes().c_str()));

    CMICmnLLDBDebugSessionInfo &rSessionInfo(CMICmnLLDBDebugSessionInfo::Instance());
    const lldb::ReturnStatus rtn = rSessionInfo.GetDebugger().GetCommandInterpreter().HandleCommand(strCmd.c_str(), m_lldbResult);
    MIunused(rtn);

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Helper function for parsing the header returned from lldb for the command:
//              target modules dump line-table <file>
//          where the header is of the format:
//              Line table for /path/to/file in `/path/to/module
// Args:    input - (R) Input string to parse.
//          file  - (W) String representing the file.
// Return:  bool - True = input was parsed successfully, false = input could not be parsed.
// Throws:  None.
//--
static bool
ParseLLDBLineAddressHeader(const char *input, CMIUtilString &file)
{
    // Match LineEntry using regex.
    static MIUtilParse::CRegexParser g_lineentry_header_regex( 
        "^ *Lines found for file (.+) in compilation unit (.+) in `(.+)$");
        //                       ^1=file                  ^2=cu    ^3=module

    MIUtilParse::CRegexParser::Match match(4);

    const bool ok = g_lineentry_header_regex.Execute(input, match);
    if (ok)
        file = match.GetMatchAtIndex(1);
    return ok;
}

//++ ------------------------------------------------------------------------------------
// Details: Helper function for parsing a line entry returned from lldb for the command:
//              target modules dump line-table <file>
//          where the line entry is of the format:
//              0x0000000100000e70: /path/to/file:3002[:4]
//              addr                file          line column(opt)
// Args:    input - (R) Input string to parse.
//          addr  - (W) String representing the pc address.
//          file  - (W) String representing the file.
//          line  - (W) String representing the line.
// Return:  bool - True = input was parsed successfully, false = input could not be parsed.
// Throws:  None.
//--
static bool
ParseLLDBLineAddressEntry(const char *input, CMIUtilString &addr,
                          CMIUtilString &file, CMIUtilString &line)
{
    // Note: Ambiguities arise because the column is optional, and
    // because : can appear in filenames or as a byte in a multibyte
    // UTF8 character.  We keep those cases to a minimum by using regex
    // to work on the string from both the left and right, so that what
    // is remains is assumed to be the filename.

    // Match LineEntry using regex.
    static MIUtilParse::CRegexParser g_lineentry_nocol_regex( 
        "^ *\\[(0x[0-9a-fA-F]+)-(0x[0-9a-fA-F]+)\\): (.+):([0-9]+)$");
    static MIUtilParse::CRegexParser g_lineentry_col_regex( 
        "^ *\\[(0x[0-9a-fA-F]+)-(0x[0-9a-fA-F]+)\\): (.+):([0-9]+):[0-9]+$");
        //     ^1=start         ^2=end               ^3=f ^4=line ^5=:col(opt)

    MIUtilParse::CRegexParser::Match match(6);

    // First try matching the LineEntry with the column,
    // then try without the column.
    const bool ok = g_lineentry_col_regex.Execute(input, match) ||
                    g_lineentry_nocol_regex.Execute(input, match);
    if (ok)
    {
        addr = match.GetMatchAtIndex(1);
        file = match.GetMatchAtIndex(3);
        line = match.GetMatchAtIndex(4);
    }
    return ok;
}

//++ ------------------------------------------------------------------------------------
// Details: The invoker requires this function. The command prepares a MI Record Result
//          for the work carried out in the Execute().
// Type:    Overridden.
// Args:    None.
// Return:  MIstatus::success - Functional succeeded.
//          MIstatus::failure - Functional failed.
// Throws:  None.
//--
bool
CMICmdCmdSymbolListLines::Acknowledge()
{
    if (m_lldbResult.GetErrorSize() > 0)
    {
        const char *pLldbErr = m_lldbResult.GetError();
        const CMIUtilString strMsg(CMIUtilString(pLldbErr).StripCRAll());
        const CMICmnMIValueConst miValueConst(strMsg);
        const CMICmnMIValueResult miValueResult("message", miValueConst);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult);
        m_miResultRecord = miRecordResult;
    }
    else
    {
        CMIUtilString::VecString_t vecLines;
        const CMIUtilString strLldbMsg(m_lldbResult.GetOutput());
        const MIuint nLines(strLldbMsg.SplitLines(vecLines));

        // Parse the file from the header.
        const CMIUtilString &rWantFile(vecLines[0]);
        CMIUtilString strWantFile;
        if (!ParseLLDBLineAddressHeader(rWantFile.c_str(), strWantFile))
        {
            // Unexpected error - parsing failed.
            // MI print "%s^error,msg=\"Command '-symbol-list-lines'. Error: Line address header is absent or has an unknown format.\""
            const CMICmnMIValueConst miValueConst(CMIUtilString::Format(MIRSRC(IDS_CMD_ERR_SOME_ERROR), m_cmdData.strMiCmd.c_str(), "Line address header is absent or has an unknown format."));
            const CMICmnMIValueResult miValueResult("msg", miValueConst);
            const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult);
            m_miResultRecord = miRecordResult;

            return MIstatus::success;
        }

        // Parse the line address entries.
        CMICmnMIValueList miValueList(true);
        for (MIuint i = 1; i < nLines; ++i)
        {
            // String looks like:
            // 0x0000000100000e70: /path/to/file:3[:4]
            const CMIUtilString &rLine(vecLines[i]);
            CMIUtilString strAddr;
            CMIUtilString strFile;
            CMIUtilString strLine;

            if (!ParseLLDBLineAddressEntry(rLine.c_str(), strAddr, strFile, strLine))
                continue;

            const CMICmnMIValueConst miValueConst(strAddr);
            const CMICmnMIValueResult miValueResult("pc", miValueConst);
            CMICmnMIValueTuple miValueTuple(miValueResult);

            const CMICmnMIValueConst miValueConst2(strLine);
            const CMICmnMIValueResult miValueResult2("line", miValueConst2);
            miValueTuple.Add(miValueResult2);

            miValueList.Add(miValueTuple);
        }

        // MI print "%s^done,lines=[{pc=\"%d\",line=\"%d\"}...]"
        const CMICmnMIValueResult miValueResult("lines", miValueList);
        const CMICmnMIResultRecord miRecordResult(m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult);
        m_miResultRecord = miRecordResult;
    }

    return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details: Required by the CMICmdFactory when registering *this command. The factory
//          calls this function to create an instance of *this command.
// Type:    Static method.
// Args:    None.
// Return:  CMICmdBase * - Pointer to a new command.
// Throws:  None.
//--
CMICmdBase *
CMICmdCmdSymbolListLines::CreateSelf()
{
    return new CMICmdCmdSymbolListLines();
}
